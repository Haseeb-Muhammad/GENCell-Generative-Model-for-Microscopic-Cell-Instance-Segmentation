import tifffile as tif
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
  

def prepare_cellTracking_data(data_dir, dest_dir=""):
    os.makedirs(dest_dir,exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "image"),exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "image", "test"),exist_ok=True)

    image_dirs = [os.path.join(data_dir,"01"), os.path.join(data_dir,"02")]

    for k in range(2):

        image_paths = sorted(glob.glob(image_dirs[k]+"/*"))
        
        for i, image_path in tqdm(enumerate(sorted(image_paths)), total=len(image_paths), desc="Enumerating"):
            parts = image_path.split('/')
            
            split="test"
            
            img = tif.imread(image_path)
            for j, image_slice in enumerate(img):
                
                # Check for first or last slice
                if (j==0) or (j==len(img)-1):
                    img_slice = np.array([image_slice,image_slice,image_slice])
                else:
                    img_slice = img[j-1:j+2]
                    
                img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
                img_slice=np.transpose(img_slice, (1,2,0))

                slice_full_name = f"{parts[-2].split('_')[0]}_{parts[-1].split('.')[0]}_slice_{str(j)}.png" 
                
                cv2.imwrite(os.path.join(dest_dir,"image",split, slice_full_name), cv2.cvtColor(img_slice, cv2.COLOR_RGB2BGR))
            

prepare_cellTracking_data(data_dir="/netscratch/muhammad/datasets/Fluo-N3DH-CHO", dest_dir="/netscratch/muhammad/ProcessedDatasets/Fluo-N3DH-CHO")
