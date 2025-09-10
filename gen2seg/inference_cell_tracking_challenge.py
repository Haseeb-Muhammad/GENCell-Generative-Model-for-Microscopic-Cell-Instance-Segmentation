import torch
from gen2seg_sd_pipeline import gen2segSDPipeline  # Import your custom pipeline
from PIL import Image
import numpy as np
from cellTrackingChallengeLoader import CellTrackingChallengeTest
from tqdm import tqdm
import os
import shutil

fluo_root = "/netscratch/muhammad/ProcessedDatasets/Fluo-C3DL-MDA231"

test_dataset_fluo = CellTrackingChallengeTest(root_dir=fluo_root, split="test")
test_dataloader   = torch.utils.data.DataLoader(
    test_dataset_fluo,
    batch_size=1
)
    
def organize_weights(weigths_dir, eval_weights_dir="/netscratch/muhammad/codes/gen2seg/training/model-finetuned/test_weights"):
    '''
        organize all weigths into "current_val_weigths" directory in args.output_dir
        
        Args:
            epoch (int) : current epoch

        Return:
            current epoch weigths path (str)

    '''
    unet_path = os.path.join(weigths_dir,"unet")
    dest_path = os.path.join(eval_weights_dir, f"unet")
    
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
        print("Current tree deleted")
    shutil.copytree(unet_path, dest_path)
    
    return eval_weights_dir

def evaluate(weight_dir, dest_dir="/netscratch/muhammad/codes/gen2seg/results/Fluo-C3DL-MDA231"):
    
    weights_path = organize_weights(weigths_dir=weight_dir)
    training_name = weight_dir.split("/")[-2]
    epoch = weight_dir.split("/")[-1]
    os.makedirs(os.path.join(dest_dir, training_name), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, training_name, epoch), exist_ok=True)
    dest_dir = os.path.join(dest_dir, training_name, epoch)
    
    pipe = gen2segSDPipeline.from_pretrained(
        weights_path,
        use_safetensors=True,         
    ).to("cuda")

    with torch.no_grad(): 
        for batch_index, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader), desc="Testing")):
            image = Image.open(batch[0]).convert("RGB")

            orig_res = image.size

            seg = pipe(image).prediction.squeeze()
            seg = np.array(seg).astype(np.uint8)
            pred = Image.fromarray(seg).resize(orig_res, Image.LANCZOS)

            image_name = os.path.basename(batch[0])
            dest_path = os.path.join(dest_dir, image_name)
            pred.save(dest_path)
            
evaluate(weight_dir="/netscratch/muhammad/codes/gen2seg/training/model-finetuned/stable_diffusion_extendedDatasetAllSlicesExtendedTraining/Epoch-5")