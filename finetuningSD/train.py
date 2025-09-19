import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from accelerate.logging import get_logger
import accelerate
import torch.nn.functional as F


from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
from PIL import Image




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_dir",
        type=str,
        default=""
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=500, 
        help="A seed for reproducible training."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means main process.",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="e2e-ft-diffusion_fluo_dataset",
        help="The `project_name` arg passed to Accelerator.init_trackers.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period).",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether to use xformers efficient attention (saves memory)."
    )

    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=2, 
        help="Batch size (per device) for the training dataloader."
    )

    args = parser.parse_args()
    return args

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def encode_image(vae, image):
    """Apply VAE Encoder to image."""
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent

def decode_image(vae, latent):
    """Apply VAE Decoder to latent."""
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image

def main():
    args = parse_args()
    
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )
    logger = get_logger(__name__, log_level="INFO")

    if accelerator.is_main_process():
        tracker_config = dict(vars(args))
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.init_trackers(args.tracker_project_name, tracker_config)

        # Load model components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer       = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder    = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae            = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet           = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        unet.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        unet.to(dtype=weight_dtype)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    fluo_root = "C:\\Users\\hasee\\Desktop\\DFKI\\processedDatasets\\Fluo-N3DH-SIM+"
    train_dataset_fluo = Fluo_N3DH_SIM(root_dir=fluo_root, split="train")

    train_dataloader   = torch.utils.data.DataLoader(
        train_dataset_fluo,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Freeze VAE and CLIP (text encoder), train only UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

        # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 has known issues on some GPUs. If you see problems, update to >=0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install xformers to use memory efficient attention.")

        # For saving/loading with accelerate >= 0.16.0
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Prepare with accelerator (move to GPU, handle DDP, etc.)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Pre-compute empty text CLIP encoding
    empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token    = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)

    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Encode RGB to latents
            
            rgb_latents = encode_image(
                vae,
                batch["rgb"].to(device=accelerator.device, dtype=weight_dtype)
            )
            rgb_latents = rgb_latents * vae.config.scaling_factor

            noise = torch.randn(rgb_latents.shape).to(rgb_latents.device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (args.train_batch_size,), device=rgb_latents.device).long()

            noisy_image = noise_scheduler.add_noise(rgb_latents, noise, timesteps)

            # UNet input: (rgb_latents, noisy_latents) if condition exists
            encoder_hidden_states = empty_encoding.repeat(len(batch["rgb"]), 1, 1)


            with accelerator.accumulate(unet):
                noise_pred = unet(
                    noisy_image,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False)[0]
                
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        save_path = os.path.join(args.output_dir, f"Epoch-{global_step}")
        accelerator.save_state(save_path)
        
            

if "__main__" == __name__:
    main()