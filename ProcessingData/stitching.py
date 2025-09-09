import argparse
import os
from PIL import Image
from utils import color_instances_from_sobel, plot_sobel_gradients, remove_noise, extract_mask, remove_noise_advanced,remove_noise_with_area
from metric_calculation import calculate_iou 
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from collections import defaultdict



def transform_slice(slice_array: np.ndarray) -> np.ndarray:
    """
    Transform a 2D image slice by applying Sobel gradient detection, coloring instances, and removing noise.

    Args:
        slice_array (np.ndarray): A RGB 2D NumPy array representing the image slice to be processed.

    Returns:
        np.ndarray: The processed image slice with noise removed after gradient and instance coloring steps.
    """
    gradients = plot_sobel_gradients(slice=slice_array)
    uniform_predictions = color_instances_from_sobel(sobel_image=gradients)
    noise_removed = remove_noise(uniform_predictions)
    return noise_removed

def process_2D_slices_generator(pred_2D_path: str, time_step: str, image_num: str, num_slices: int = 59):
    """
    Generator that yields processed 2D slices one at a time to save memory.
    
    Args:
        pred_2D_path (str): Path to the 2D predictions directory
        time_step (str): time step of the image e.g 01
        image_num (str): image number e.g 061
        num_slices (int): number of slices in the image
        
    Yields:
        np.ndarray: Processed 2D slice
    """
    for slice_num in range(num_slices):
        slice_name = f"{time_step}_man_seg{image_num}_slice_{slice_num}.png"
        slice_path = os.path.join(pred_2D_path, slice_name)
        
        # Load and process slice
        with Image.open(slice_path) as img:
            slice_array = np.array(img)
        
        processed_slice = transform_slice(slice_array)
        yield processed_slice

def get_instance_masks_optimized(slice_array: np.ndarray) -> dict:
    """
    Efficiently extract all instance masks from a slice using vectorized operations.
    
    Args:
        slice_array (np.ndarray): 2D array containing instance labels
        
    Returns:
        dict: {instance_id: mask} where mask is boolean array
    """
    unique_instances = np.unique(slice_array)
    unique_instances = unique_instances[unique_instances != 0]  # Remove background
    
    masks = {}
    for instance_id in unique_instances:
        masks[instance_id] = slice_array == instance_id
    
    return masks

def calculate_iou_vectorized(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Vectorized IoU calculation for boolean masks.
    
    Args:
        mask1, mask2 (np.ndarray): Boolean masks
        
    Returns:
        float: IoU value
    """
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0

def process_3D_slices_streaming(slice_generator, iou_threshold: float = 0.0):
    """
    Process 3D slices with streaming approach to minimize memory usage.
    
    Args:
        slice_generator: Generator yielding 2D slices
        iou_threshold (float): Minimum IoU threshold for matching instances
        
    Returns:
        np.ndarray: 3D array with consistent instance IDs across slices
    """
    # Use defaultdict for cleaner code
    global_instances = defaultdict(dict)  # {global_id: {slice_idx: local_id}}
    global_instance_id = 1
    
    # Store only previous slice masks to save memory
    prev_masks = {}
    prev_to_global = {}  # {prev_local_id: global_id}
    
    processed_slices = []
    
    for slice_idx, current_slice in enumerate(tqdm(slice_generator, desc="Processing slices")):
        current_masks = get_instance_masks_optimized(current_slice)
        slice_mapping = {}  # {local_id: global_id}
        
        if slice_idx == 0:
            # First slice: assign global IDs directly
            for local_id in current_masks:
                global_instances[global_instance_id][slice_idx] = local_id
                slice_mapping[local_id] = global_instance_id
                global_instance_id += 1
        else:
            # Find best matches with previous slice
            used_global_ids = set()
            
            # Pre-compute all IoU values for efficiency
            iou_matrix = {}
            for curr_id, curr_mask in current_masks.items():
                for prev_id, prev_mask in prev_masks.items():
                    iou = calculate_iou_vectorized(curr_mask, prev_mask)
                    if iou > iou_threshold:
                        iou_matrix[(curr_id, prev_id)] = iou
            
            # Sort by IoU descending for greedy matching
            sorted_matches = sorted(iou_matrix.items(), key=lambda x: x[1], reverse=True)
            
            # Assign matches greedily
            used_curr_ids = set()
            used_prev_ids = set()
            
            for (curr_id, prev_id), iou in sorted_matches:
                if curr_id in used_curr_ids or prev_id in used_prev_ids:
                    continue
                
                # Found valid match
                if prev_id in prev_to_global:
                    global_id = prev_to_global[prev_id]
                    global_instances[global_id][slice_idx] = curr_id
                    slice_mapping[curr_id] = global_id
                    used_curr_ids.add(curr_id)
                    used_prev_ids.add(prev_id)
                    used_global_ids.add(global_id)
            
            # Create new instances for unmatched current instances
            for curr_id in current_masks:
                if curr_id not in used_curr_ids:
                    global_instances[global_instance_id][slice_idx] = curr_id
                    slice_mapping[curr_id] = global_instance_id
                    global_instance_id += 1
        
        # Create remapped slice
        remapped_slice = np.zeros_like(current_slice, dtype=np.uint16)  # Use uint16 for more instances
        for local_id, global_id in slice_mapping.items():
            remapped_slice[current_slice == local_id] = global_id
        
        processed_slices.append(remapped_slice)
        
        # Update for next iteration (only keep what we need)
        prev_masks = current_masks
        prev_to_global = {local_id: global_id for local_id, global_id in slice_mapping.items()}
    
    return np.array(processed_slices, dtype=np.uint16)

def format_image_number(image_num: int) -> str:
    """Helper function to format image number with leading zeros."""
    return f"{image_num:03d}"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_2D",
        type=str,
        default="/netscratch/muhammad/codes/gen2seg/results/Fluo-N3DH-SIM+/2DResults/stable_diffusion_extendedDatasetAllSlicesExtendedTraining_finetuning/Epoch-5"
    )

    parser.add_argument(
        "--root_directory",
        type=str,
        default="/netscratch/muhammad/codes/gen2seg/results/Fluo-N3DH-SIM+/3DResults/stable_diffusion_extendedDatasetAllSlicesExtendedTraining_finetuning/Epoch-5"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    time_steps = ["01","02"]
    image_ranges = [range(131,150), range(61,80)]

    for time_step, image_range in zip(time_steps, image_ranges):
        dir_name = f"{time_step}_RES"
        output_dir = os.path.join(args.root_directory, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        for image_num in tqdm(image_range, desc=f"Processing images for {time_step}"):
            image_num_str = format_image_number(image_num)
            
            # Use generator for memory-efficient processing
            slice_gen = process_2D_slices_generator(
                pred_2D_path=args.pred_2D, 
                time_step=time_step, 
                image_num=image_num_str, 
                num_slices=59
            )
            
            # Process with streaming approach
            result_3d = process_3D_slices_streaming(slice_gen)
            
            # Save result
            pred_3d_name = f"mask{image_num_str}.tif"
            output_path = os.path.join(output_dir, pred_3d_name)
            tiff.imwrite(output_path, result_3d, photometric="minisblack")

if __name__ == "__main__":
    main()