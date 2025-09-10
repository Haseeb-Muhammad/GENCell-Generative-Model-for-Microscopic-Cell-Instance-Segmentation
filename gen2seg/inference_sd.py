import torch
from gen2seg_sd_pipeline import gen2segSDPipeline  # Import your custom pipeline
from PIL import Image
import numpy as np
import time

# Load the image
image_path =  '/netscratch/muhammad/ProcessedDatasets/Fluo-N3DH-SIM+/image/test/01_man_seg137_slice_25.png'
image = Image.open(image_path).convert("RGB")
orig_res = image.size
output_path = "/netscratch/muhammad/codes/gen2seg/results/01_man_seg137_slice_25_epoch_3_withVariationalLossOnly.png" 

pipe = gen2segSDPipeline.from_pretrained(
    "/netscratch/muhammad/codes/gen2seg/training/model-finetuned/val_weights",
    use_safetensors=True,         # Use safetensors if available
).to("cuda")  # Ensure the pipeline is moved to CUDA

# Load the pipeline and generate the segmentation map
with torch.no_grad():

    start_time = time.time()
    # Generate segmentation map
    seg = pipe(image).prediction.squeeze()
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")

seg = np.array(seg).astype(np.uint8)
Image.fromarray(seg).resize(orig_res, Image.LANCZOS).save(output_path)
print(f"Saved output image to {output_path}")