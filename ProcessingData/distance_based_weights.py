import argparse
import glob
import json
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

def rgb_to_instance_id(rgb_pixel: np.ndarray) -> int:
    """Convert RGB pixel to instance ID."""
    if np.all(rgb_pixel == 0):
        return 0
    return int(rgb_pixel[0]) * 256 * 256 + int(rgb_pixel[1]) * 256 + int(rgb_pixel[2])


def instance_id_to_rgb(instance_id: int) -> np.ndarray:
    """Convert instance ID to RGB pixel."""
    if instance_id == 0:
        return np.array([0, 0, 0])
    r = instance_id // (256 * 256)
    g = (instance_id % (256 * 256)) // 256
    b = instance_id % 256
    return np.array([r, g, b])


def instance_distance(gt: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate the distance of all instances from all other instances.

    Args:
        gt (np.ndarray): Ground truth image as numpy array

    Returns:
        Dict[str, Dict[str, float]]: {instanceID: {instanceID: distance_between_instances}}
    """
    rows, cols = gt.shape[:2]

    # Build instance ID map
    instance_map = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            instance_map[i, j] = rgb_to_instance_id(gt[i, j])

    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]

    # Calculate centroid coordinates for each instance
    inst_coords = {}  # {inst_id: (coord_x, coord_y), ...}
    print(f"{instance_ids=}")
    for instance_id in instance_ids:
        coords = np.argwhere(np.where(instance_map == instance_id, 1, 0))
        
        mean_x, mean_y = np.mean(coords, axis=0)
        mean_x, mean_y = int(mean_x), int(mean_y)
        
        inst_coords[instance_id] = (mean_x, mean_y)
    
    # Calculate distances between all pairs of instances
    distances = {}  # {inst_id: {inst_id: dist, ...}, ...}
    for inst, coord in inst_coords.items():
        distances[str(inst)] = {}
        for inst1, coord1 in inst_coords.items():
            if inst1 != inst:
                distances[str(inst)][str(inst1)] = np.sqrt(
                    (coord1[0] - coord[0])**2 + (coord1[1] - coord[1])**2
                )
    
    print("Distance calculation completed")
    return distances

def calculate_exponential_weights(distances_dict: Dict[str, Dict[str, float]], 
                                decay_factor: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    Calculate weights using exponential decay for instance segmentation loss.
    
    Closer instances get higher weights using the formula:
    weight = exp(-decay_factor × normalized_distance)
    
    Args:
        distances_dict: {instance_id_str: {neighbor_id_str: distance}}
        decay_factor: Controls weight decay rate. Higher = faster decay.
                     Recommended values: 0.5-2.0
    
    Returns:
        Dict: {instance_id_str: {neighbor_id_str: weight}} where weights ∈ (0, 1]
    """
    if not distances_dict:
        return {}
    
    # Collect all distances for normalization
    all_distances = []
    for neighbors in distances_dict.values():
        all_distances.extend(neighbors.values())
    
    if not all_distances:
        return {}
    
    min_distance = min(all_distances)
    max_distance = max(all_distances)
    distance_range = max_distance - min_distance

    # Calculate weights for each instance-neighbor pair
    weights_dict = {}
    for instance_id, neighbors in distances_dict.items():
        weights_dict[str(instance_id)] = {}
        
        for neighbor_id, distance in neighbors.items():
            # Normalize distance to [0, 1] range
            if distance_range == 0:
                normalized_distance = 0.0
            else:
                normalized_distance = (distance - min_distance) / distance_range
            
            # Apply exponential decay: closer = higher weight
            weight = np.exp(-decay_factor * normalized_distance)
            weights_dict[str(instance_id)][str(neighbor_id)] = weight
    
    return weights_dict


def apply_weights_to_image(image_path: str, decay_factor: float = 5.0) -> Dict[str, Dict[str, float]]:
    """
    Complete pipeline: Load image → Calculate distances → Generate weights.
    
    Args:
        image_path: Path to RGB instance segmentation image
        decay_factor: Exponential decay parameter
        
    Returns:
        Dict: Weights dictionary ready for loss calculation
    """
    # Load and process image
    img = np.array(Image.open(image_path))
    
    # Calculate distances between all instances
    distances = instance_distance(img)
    
    # Generate exponential decay weights
    weights = calculate_exponential_weights(distances, decay_factor)
    
    # Clear intermediate variables to free memory
    del img
    del distances
    
    return weights


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="/netscratch/muhammad/ProcessedDatasets/Fluo-N3DH-SIM+",
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--dest_path", 
        type=str, 
        default="neighbors.json",
        help="Destination path for the output file"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function to process images and generate distance-based weights."""
    args = parse_args()
    
    for split in ["train"]:
        gt_paths = sorted(glob.glob(os.path.join(args.root_dir, "gt", split, "*.png")))

        neighbors = {}
        batch_size = 100  # Process in batches to avoid memory overflow
        
        for i, gt_path in enumerate(tqdm(gt_paths)):
            neighbors_dict = apply_weights_to_image(image_path=gt_path)
            neighbors[gt_path] = neighbors_dict
            
            # Save and clear memory every batch_size images
            if (i + 1) % batch_size == 0 or (i + 1) == len(gt_paths):
                # Load existing data if file exists
                output_file = os.path.join(args.root_dir, "gt", f"{split}_distances.json")
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        existing_neighbors = json.load(f)
                    existing_neighbors.update(neighbors)
                    neighbors = existing_neighbors
                
                # Save updated data
                with open(output_file, "w") as f:
                    json.dump(neighbors, f, indent=4)
                
                # Clear memory
                neighbors = {}
                print(f"Processed and saved batch up to image {i+1}/{len(gt_paths)}")

if __name__ == "__main__":
    main()
