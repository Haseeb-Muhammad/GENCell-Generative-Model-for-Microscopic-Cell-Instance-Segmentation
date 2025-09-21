# gen2seg official inference pipeline code for Stable Diffusion model
#
# This code was adapted from Marigold and Diffusion E2E Finetuning.
#
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper, along with the two works above. 

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys

import accelerate
import datasets
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from sklearn.metrics import jaccard_score
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

# Custom modules
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from gen2seg_sd_pipeline import gen2segSDPipeline

from dataloaders.load import *
from util.lr_scheduler import IterExponential
from util.neighboringloss import NeighboringLoss
from util.noise import pyramid_noise_like
from util.unet_prep import replace_unet_conv_in

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# Constants
VALIDATION_BATCH_LIMIT = 8
CHECKPOINT_FREQUENCY = 8
VALIDATION_LOG_IMAGE_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
IMAGE_RESOLUTION = (480, 640)

#############
# Arguments
#############

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'."
    )

    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_total_iter_length", type=int, default=20000)

    # Stable diffusion training settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/model-finetuned",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=500, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=2, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means main process.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "TensorBoard log directory. Will default to output_dir/runs/CURRENT_DATETIME_HOSTNAME."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20000,
        help="Save a checkpoint of the training state every X updates (suitable for resume).",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from a previous checkpoint path, or 'latest' to auto-select the last checkpoint.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether to use xformers efficient attention (saves memory)."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="e2e-ft-diffusion_fluo_dataset",
        help="The `project_name` arg passed to Accelerator.init_trackers.",
    )
    parser.add_argument(
        "--random_state_file",
        type=str,
        default=None,
        help="Used to load a random_states_0.pkl to ensure consistency across runs. "
    )
    parser.add_argument("--noise_type", type=str, default=None, help="Type of noise to apply.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


########################
# VAE Helper Functions
########################

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
        

##########################
# MAIN Training Function
##########################

def main():
    args = parse_args()

    # Saving images from prediction
    dest_dir = os.path.join(args.output_dir, "prediction")
    os.makedirs(dest_dir, exist_ok=True)

    # Transformations for validation
    to_tensor = transforms.ToTensor()
    instance_loss = NeighboringLoss()
    resize_nn = transforms.Resize(IMAGE_RESOLUTION, interpolation=Image.NEAREST)
    resize = transforms.Resize(IMAGE_RESOLUTION, interpolation=Image.BILINEAR)
        
    def organize_weights(epoch):
        """
        Organize all weights into "current_val_weights" directory in args.output_dir
        
        Args:
            epoch (int): Current epoch

        Returns:
            str: Current epoch weights path
        """

        unet_path = os.path.join(args.output_dir, f"Epoch-{epoch}","unet")
        val_weights_path = os.path.join(os.path.dirname(args.output_dir), "val_weights")

        dest_path = os.path.join(val_weights_path, f"unet")
        
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
            print("Current tree deleted")
        shutil.copytree(unet_path, dest_path)
        
        return val_weights_path


    def validation(epoch, global_step):
        """
        Performs validation and logs the results

        Args:
            epoch (int): Current epoch
            global_step (int): Current global step
        """

        val_loss = torch.tensor(0.0, requires_grad=False)
        instance_loss.current_intra_loss = 0
        instance_loss.current_inter_instance = 0
        instance_loss.current_mean_loss = 0

        weights_path = organize_weights(epoch)
        
        pipe = gen2segSDPipeline.from_pretrained(
            weights_path,
            use_safetensors=True,         
        ).to("cuda")

        with torch.no_grad(): 
            for batch_index, batch in enumerate(tqdm(val_dataloader, total=len(val_dataloader), desc="Validating")):
                image = ((batch["rgb"] + 1) / 2)
                seg = pipe(image).prediction

                val_step = (epoch * 10000) + batch_index
                ground_truth = batch["instance"]

                if batch_index in VALIDATION_LOG_IMAGE_INDICES:
                    print("Logging validation images")

                    accelerator.log({"val_prediction": wandb.Image(seg[0]),
                                    "val_Image": wandb.Image(batch["img_path"][0]),
                                    "val_GT": wandb.Image(batch["gt_path"][0]),
                                    "val_step": val_step
                                    })            
                    
                # Transformation for loss
                seg = np.squeeze(seg)
                seg = to_tensor(seg)
                seg = resize(seg)
                seg = seg[None, :, :, :]

                estimation_loss = instance_loss(
                    seg.to("cuda"), 
                    ground_truth.to("cuda"), 
                    batch["no_bg"], 
                    neighbors=batch["neighbors"]
                )
                val_loss = val_loss + estimation_loss 

                # if batch_index > VALIDATION_BATCH_LIMIT:
                #     break    
            mean_val_loss = val_loss / len(val_dataloader)
            accelerator.log({"validation_loss": mean_val_loss,
                            "global_step": global_step,
                            "val Intra-Instance Variance Loss": instance_loss.current_intra_loss / len(val_dataloader),
                            "val Inter-Instance Separation Loss": instance_loss.current_inter_instance / len(val_dataloader),
                            "val Mean-Level Separation Loss": instance_loss.current_mean_loss / len(val_dataloader)
                        })
            
            instance_loss.current_intra_loss = 0
            instance_loss.current_inter_instance = 0
            instance_loss.current_mean_loss = 0
            
            del seg
            del ground_truth
            del image
            
            del estimation_loss
            del pipe



    def save_prediction(prediction_batch, source_path_batch, batch_no):
        """
        Save prediction from decoder based on batch number
        
        Args:
            prediction_batch (tensor): Predicted masks (B, C, H, W)
            source_path_batch (list[str]): List of source images
            batch_no (int): Batch number
        """

        dest_path = os.path.join(dest_dir, str(batch_no))
        os.makedirs(dest_path, exist_ok=True)

        for i in range(args.train_batch_size):
            image_name = os.path.basename(source_path_batch[i])
            save_path = os.path.join(dest_path, image_name) 
            save_image(prediction_batch[i] / 255, save_path)


    # Init accelerator and logger
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    if args.random_state_file is not None:
        logger.info(f"Loading random states from file {args.random_state_file}")
        random_states = torch.load(args.random_state_file)
        import random
        # Python's built-in random
        random.setstate(random_states["random_state"])
        # NumPy
        np.random.set_state(random_states["numpy_random_seed"])
        # Torch CPU
        torch.set_rng_state(random_states["torch_manual_seed"])
        # Torch CUDA (for each visible GPU)
        if "torch_cuda_manual_seed" in random_states:
            for i, cuda_state in enumerate(random_states["torch_cuda_manual_seed"]):
                if torch.cuda.device_count() > i:
                    torch.cuda.set_rng_state(cuda_state, device=i)
    # Save training arguments in a .txt file
    if accelerator.is_main_process:

        args_dict = vars(args)
        args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, 'w') as file:
            file.write(args_str)

    if args.noise_type is None:
        logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning (Stable Diffusion).")

    # Load model components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer       = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder    = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae            = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet           = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None)

    # Modify UNet input if noise_type is not None
    if args.noise_type is not None:
        if unet.config['in_channels'] != 8:
            replace_unet_conv_in(unet, repeat=2)
            logger.info("Unet conv_in layer replaced for (RGB + condition) input")

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

    # For saving/loading with accelerate >= 0.16.0from gen2seg_sd_pipeline import gen2segSDPipeline
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

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_func      = IterExponential(
        total_iter_length=args.lr_total_iter_length * accelerator.num_processes,
        final_ratio=0.05,
        warmup_steps=args.lr_exp_warmup_steps * accelerator.num_processes
    )
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Custom collate function to handle neighbors field
    def custom_collate_fn(batch):
        """
        Custom collate function to handle batching when samples have different neighbor dictionaries.
        """
        # Separate the different types of data
        rgb_tensors = []
        instance_tensors = []
        no_bg_values = []
        img_paths = []
        gt_paths = []
        neighbors_list = []
        
        for sample in batch:
            rgb_tensors.append(sample['rgb'])
            instance_tensors.append(sample['instance'])
            no_bg_values.append(sample['no_bg'])
            img_paths.append(sample['img_path'])
            gt_paths.append(sample['gt_path'])
            neighbors_list.append(sample['neighbors'])
        
        # Stack tensors normally
        rgb_batch = torch.stack(rgb_tensors)
        instance_batch = torch.stack(instance_tensors)
        no_bg_batch = torch.tensor(no_bg_values)
        
        return {
            'rgb': rgb_batch,
            'instance': instance_batch, 
            'no_bg': no_bg_batch,
            'img_path': img_paths,
            'gt_path': gt_paths,
            'neighbors': neighbors_list  # Keep as list of dicts
        }

    # Setup datasets / dataloaders
    # EDIT THESE PATHS
    fluo_root_dir = "/netscratch/muhammad/ProcessedDatasets/Fluo-N3DH-SIM+"
    train_dataset_fluo = Fluo_N3DH_SIM_with_neighbors(root_dir=fluo_root_dir, split="train")
    val_dataset_fluo = Fluo_N3DH_SIM_with_neighbors(root_dir=fluo_root_dir, split="val")

    train_dataloader   = torch.utils.data.DataLoader(
        train_dataset_fluo,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_fluo,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate_fn
    )

    # Prepare with accelerator (move to GPU, handle DDP, etc.)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Choose weight dtype for model
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        unet.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        unet.to(dtype=weight_dtype)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Compute number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps*accelerator.num_processes))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        wandb_tracker = accelerator.get_tracker("wandb").tracker
        wandb_tracker.define_metric(name="val_Image", step_metric="val_step")
        wandb_tracker.define_metric(name="val_GT", step_metric="val_step")
        wandb_tracker.define_metric(name="val_prediction", step_metric="val_step")
        wandb_tracker.define_metric(name="val Intra-Instance Variance Loss by step",step_metric="val_step")
        wandb_tracker.define_metric(name="val Inter-Instance Separation Loss by step",step_metric="val_step")
        wandb_tracker.define_metric(name="val Mean-Level Separation Loss by step",step_metric="val_step")
        wandb_tracker.define_metric(name="validation_loss by step", step_metric="val_step")
        wandb_tracker.define_metric(name="IoU by step", step_metric="val_step")

        wandb_tracker.define_metric(name="Intra-Instance Variance Loss",step_metric="global_step")
        wandb_tracker.define_metric(name="Inter-Instance Separation Loss",step_metric="global_step")
        wandb_tracker.define_metric(name="Mean-Level Separation Loss",step_metric="global_step")
        wandb_tracker.define_metric(name="train_loss",step_metric="global_step")
        wandb_tracker.define_metric(name="lr",step_metric="global_step")
        wandb_tracker.define_metric(name="Image",step_metric="global_step")
        wandb_tracker.define_metric(name="GT",step_metric="global_step")
        wandb_tracker.define_metric(name="prediction",step_metric="global_step")
        wandb_tracker.define_metric(name="validation_loss", step_metric="global_step")
        wandb_tracker.define_metric(name="IoU", step_metric="global_step")
        wandb_tracker.define_metric(name="val Intra-Instance Variance Loss",step_metric="global_step")
        wandb_tracker.define_metric(name="val Inter-Instance Separation Loss",step_metric="global_step")
        wandb_tracker.define_metric(name="val Mean-Level Separation Loss",step_metric="global_step")



    # Function to unwrap model if compiled
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Logging info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume training if needed
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))  # sort by step
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    instance_loss     = NeighboringLoss()

    # Pre-compute empty text CLIP encoding
    empty_text_tokens = tokenizer(
        [""], 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).input_ids
    empty_text_tokens = empty_text_tokens.to(accelerator.device)
    empty_text_encoding = text_encoder(empty_text_tokens, return_dict=False)[0]
    empty_text_encoding = empty_text_encoding.to(accelerator.device)

    # For converting from predicted noise -> latents
    alpha_product = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_product = 1 - alpha_product

    log_file_path = os.path.join(args.output_dir, "output.json")

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Epoch {epoch} / {args.num_train_epochs}")
        # Optionally clear cache at epoch start
        torch.cuda.empty_cache()
        gc.collect()
        train_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc="training")):
            with accelerator.accumulate(unet):
                 
                # Checking images at spikes
                image_paths = batch["img_path"]
                image_log = {
                    'step': global_step + 1,
                    'paths': [str(p) for p in image_paths]
                }
                with open(log_file_path, 'a') as f:
                    f.write(json.dumps(image_log) + "\n")

                # Encode RGB to latents
                rgb_latents = encode_image(
                    vae,
                    batch["rgb"].to(device=accelerator.device, dtype=weight_dtype)
                )
                rgb_latents = rgb_latents * vae.config.scaling_factor  # Scale latents to unit variance

                # Timesteps - last time step supposed to be maximum noise but zero tensor is passed
                timesteps = (
                    torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) *
                    (noise_scheduler.config.num_train_timesteps - 1)
                )
                timesteps = timesteps.long()
                noisy_latents = torch.zeros_like(rgb_latents)

                # UNet input: (rgb_latents, noisy_latents) if condition exists
                encoder_hidden_states = empty_text_encoding.repeat(len(batch["rgb"]), 1, 1)

                unet_input = rgb_latents

                model_prediction = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False
                )[0]

                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                alpha_product_t = alpha_product[timesteps].view(-1, 1, 1, 1)
                beta_product_t = beta_product[timesteps].view(-1, 1, 1, 1)

                if noise_scheduler.config.prediction_type == "v_prediction":
                    current_latent_estimate = (
                        (alpha_product_t**0.5) * noisy_latents - 
                        (beta_product_t**0.5) * model_prediction
                    )
                elif noise_scheduler.config.prediction_type == "epsilon":
                    current_latent_estimate = (
                        (noisy_latents - beta_product_t**0.5 * model_prediction) / 
                        (alpha_product_t**0.5)
                    )
                elif noise_scheduler.config.prediction_type == "sample":
                    current_latent_estimate = model_prediction
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                current_latent_estimate = current_latent_estimate / vae.config.scaling_factor
                current_estimate = decode_image(vae, current_latent_estimate)

                min_val = torch.abs(current_estimate.min())
                max_val = torch.abs(current_estimate.max())
                current_estimate = (current_estimate + min_val) / (max_val + min_val + 1e-5)

                current_estimate = current_estimate * 255.0
                
                ground_truth = batch["instance"].to(device=accelerator.device, dtype=weight_dtype)


                estimation_loss = instance_loss(
                    current_estimate, 
                    ground_truth, 
                    batch["no_bg"], 
                    neighbors=batch["neighbors"]
                )
                loss = loss + estimation_loss

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Detach and log
            avg_loss = accelerator.gather(loss.detach()).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            instance_loss.current_intra_loss = instance_loss.current_intra_loss / args.gradient_accumulation_steps 
            instance_loss.current_inter_instance = instance_loss.current_inter_instance / args.gradient_accumulation_steps 
            instance_loss.current_mean_loss = instance_loss.current_mean_loss / args.gradient_accumulation_steps

            # If we just finished an accumulation step...
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging losses and resetting them
                print("Logging Training loss")
                accelerator.log(
                    {
                        "Intra-Instance Variance Loss": instance_loss.current_intra_loss,
                        "Inter-Instance Separation Loss": instance_loss.current_inter_instance,
                        "Mean-Level Separation Loss": instance_loss.current_mean_loss,
                        "train_loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "Image": wandb.Image(batch["img_path"][0], caption="Image"),
                        "GT": wandb.Image(batch["gt_path"][0], caption="GT"),
                        "prediction": wandb.Image(current_estimate[0], caption="Prediction"),
                        "global_step": global_step
                    }
                )

                instance_loss.current_intra_loss = 0
                instance_loss.current_inter_instance = 0
                instance_loss.current_mean_loss = 0
                train_loss = 0.0
               
                # Checkpoint saving
                if global_step % args.checkpointing_steps == 0:
                    logger.info(f"Saving checkpoint at step {global_step}")
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            # Remove older checkpoints if exceeding limit
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)} oldest"
                                )
                                logger.info(f"Removing: {', '.join(removing_checkpoints)}")
                                for rm_ckpt in removing_checkpoints:
                                    rm_ckpt_path = os.path.join(args.output_dir, rm_ckpt)
                                    shutil.rmtree(rm_ckpt_path)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    

            # Show step loss in progress bar
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # Delete / free memory
            del rgb_latents, noisy_latents, model_prediction
            if 'current_latent_estimate' in locals():
                del current_latent_estimate
            if 'current_estimate' in locals():
                del current_estimate
            if 'ground_truth' in locals():
                del ground_truth
            del loss

            # Early stopping
            if global_step >= args.max_train_steps:
                print(f"Breaking at {global_step=} because it is greater than {args.max_train_steps=}")
                break

            # if (step % CHECKPOINT_FREQUENCY == 0) and (step != 0):
            #     print(f"Breaking at {step=}")
            #     break

        save_path = os.path.join(args.output_dir, f"Epoch-{epoch}")
        accelerator.save_state(save_path)
        logger.info(f"Saved Epoch Weight to {save_path}")        
        
        validation(epoch, global_step)

    # Post-training: create pipeline and save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            timestep_spacing="trailing",
            revision=args.revision,
            variant=args.variant
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            variant=args.variant,
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)

    logger.info("Finished training.")
    accelerator.end_training()

if __name__ == "__main__":
    main()
