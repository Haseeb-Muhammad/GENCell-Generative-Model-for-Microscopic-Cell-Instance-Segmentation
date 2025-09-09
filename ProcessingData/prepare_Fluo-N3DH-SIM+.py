import tifffile as tif
import numpy as np
import os
import cv2
import glob
import colorsys 
from tqdm import tqdm


def generate_distinct_colors(uniq):
    n = len(uniq)
    max = np.max(uniq)
    colors = np.zeros((max+1,3))
    # print(f"{n} unqiue colors")
    for i in range(1,n): 
        hue = i / n  # evenly spaced hues 
        lightness = 0.5  
        saturation = 1.0  
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation) 
        rgb = tuple(int(c * 255) for c in rgb) 
        colors[uniq[i]] = (rgb) 
    return colors 

def convert_gt_to_rgb(slice, unique):
    colors = generate_distinct_colors(unique)
    rgb_slice = colors[slice].astype(np.uint8)
    bgr_image = cv2.cvtColor(rgb_slice, cv2.COLOR_RGB2BGR)
    return bgr_image    

def prepare_Fluo_N3DH_SIM_data(data_dir, dest_dir=""):
    # os.makedirs(dest_dir,exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "gt"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "image"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "gt", "train"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "gt", "val"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "gt", "test"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "image", "train"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "image", "val"),exist_ok=True)
    # os.makedirs(os.path.join(dest_dir, "image", "test"),exist_ok=True)

    gt_dirs = [os.path.join(data_dir,"train","01_GT", "SEG"), os.path.join(data_dir,"train","02_GT", "SEG")]
    image_dirs = [os.path.join(data_dir,"train","01"), os.path.join(data_dir,"train","02")]

    Complete_black_count = {
        "train":0,
        "val":0,
        "test":0
    }
    for k in range(2):
        if k==0:
            train,val = 120,130
        else:
            train,val = 50,60

        image_paths = sorted(glob.glob(image_dirs[k]+"/*"))
        gt_paths = sorted(glob.glob(gt_dirs[k] + "/*"))
        
        for i, gt_path in tqdm(enumerate(sorted(gt_paths)), total=len(gt_paths), desc="Enumerating"):
            parts = gt_path.split('/')
            
            if i<=train:
                split="train"
            elif i>train and i<=val:
                split="val"
            else:
                split="test"
            
            # img = tif.imread(image_paths[i])
            gt = tif.imread(gt_path)

            for j, gt_slice in enumerate(gt):

                #Checking if mask is completely black
                unique = np.unique(gt_slice) 
                if len(unique) ==1:
                    # print("Only black mask")
                    # bgr_gt = cv2.cvtColor(gt_slice, cv2.COLOR_RGB2BGR)
                    Complete_black_count[split] +=1
                # else:
                #     bgr_gt = convert_gt_to_rgb(gt_slice, unique=unique)

                # Check for first or last slice
                if (j==0) or (j==len(gt)-1):
                    # print("Last or first slice")
                    img_slice = np.array([img[j],img[j],img[j]])
                else:
                    img_slice = img[j-1:j+2]
                    
                img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
                img_slice=np.transpose(img_slice, (1,2,0))

                slice_full_name = f"{parts[-3].split('_')[0]}_{parts[-1].split('.')[0]}_slice_{str(j)}.png" 
                
                cv2.imwrite(os.path.join(dest_dir,"image",split, slice_full_name), cv2.cvtColor(img_slice, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dest_dir,"gt",split, slice_full_name), bgr_gt)
            # print(f"{gt_path=}")
    for key, value in Complete_black_count.items():
        print(f"Black count {key}: {value}")

prepare_Fluo_N3DH_SIM_data(data_dir="/netscratch/muhammad/datasets/Fluo-N3DH-SIM+", dest_dir="/netscratch/muhammad/ProcessedDatasets/Complete_Fluo-N3DH-SIM+")
