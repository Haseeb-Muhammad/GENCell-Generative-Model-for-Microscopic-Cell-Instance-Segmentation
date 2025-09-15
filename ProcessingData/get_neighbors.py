import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os
import glob
import json

def euclidean_distance(cord1,cord2):
        return (((cord1[0]-cord2[0])**2)+((cord1[1]-cord2[1])**2))**(0.5)

def rgb_to_instance_id(rgb_pixel):
    """Convert RGB pixel to unique instance ID"""
    if np.all(rgb_pixel == 0):  # Background is black (0,0,0)
        return 0
    # Convert RGB to a unique integer ID
    return int(rgb_pixel[0]) * 256 * 256 + int(rgb_pixel[1]) * 256 + int(rgb_pixel[2])

def instance_id_to_rgb(instance_id):
    """Convert instance ID back to RGB"""
    if instance_id == 0:
        return np.array([0, 0, 0])
    r = instance_id // (256 * 256)
    g = (instance_id % (256 * 256)) // 256
    b = instance_id % 256
    return np.array([r, g, b])

def prune_neighbors(neighbors_dict):
    """
    Prune neighbors for each instance based on distance using the IQR method.

    For each instance, this function removes neighbors whose distance is considered an outlier
    (greater than Q3 + 1.5 * IQR, where Q3 is the 75th percentile and IQR is the interquartile range).
    The function returns a new dictionary with only the pruned neighbors.

    Args:
        neighbors_dict (dict): A dictionary mapping instance IDs to a dictionary of neighbor instance IDs and their distances.
            Example: {instance_id: {neighbor_id: distance, ...}, ...}

    Returns:
        dict: A dictionary with the same structure as `neighbors_dict`, but with outlier neighbors removed.
    """
    pruned = 0
    pruned_neighbors = {}
    for instance_id, neighbors in neighbors_dict.items():
        pruned_neighbors[instance_id] = {}
        if not neighbors:
            continue
        distances = np.array(list(neighbors.values()))
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        upper = q3 + 0.75 * iqr
        for neighbor, distance in neighbors.items():
            if distance < upper:
                pruned_neighbors[instance_id][neighbor] = distance
            else:
                pruned += 1
    # print(f"{pruned=}")
    return pruned_neighbors

def get_neighbors_with_search(gt: np.ndarray) -> dict[int:list[int]]:
    '''
        Returns a dictionary that map instance id to an array of instances ids that are the instance's neighbours

        Args: 
            gt (np.ndarray) : represent the gt as a numpy array (H, W, 3) with unique RGB values for each instance
        
        Returns:
            Dict{int:List[int]} : key is instance id and list is list of instance ids that are it's neighbours
    '''
    
    # Get dimensions
    rows, columns = gt.shape[:2]
    
    # Create instance ID map from RGB image
    instance_map = np.zeros((rows, columns), dtype=np.int32)
    for i in range(rows):
        for j in range(columns):
            instance_map[i, j] = rgb_to_instance_id(gt[i, j])
    
    # Get unique instance IDs
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]  # Remove Background
    # print(f"Found {len(instance_ids)} unique instances")

    def search_cord(cord, current_instance):
        neighbours = {}
        directions = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(-1,1),(1,-1)]

        for dir in directions:
            new_cord = (cord[0]+dir[0], cord[1]+dir[1])
            if new_cord[0]>=rows or new_cord[1]>=columns or new_cord[0]<0 or new_cord[1]<0:
                continue
            
            pixel_instance_id = instance_map[new_cord[0], new_cord[1]]
            
            if pixel_instance_id == 0:  # Background pixel
                new_x, new_y = new_cord[0], new_cord[1]
                while new_x < (rows-1) and new_y < (columns-1) and new_x>=0 and new_y>=0:
                    new_x += dir[0]
                    new_y += dir[1]
                    next_instance_id = instance_map[new_x, new_y]
                    if next_instance_id != 0 and next_instance_id != current_instance:
                        neighbours[next_instance_id] = euclidean_distance(cord,(new_x,new_y))
                        break
            elif pixel_instance_id != current_instance:  # Different instance
                neighbours[pixel_instance_id]=1
        
        return neighbours    

    neighbors = {} # {instID : {neighbor_instance_id : dist}}
    for instance in instance_ids:
        # Find all coordinates for this instance
        cords = np.argwhere(instance_map == instance)
        neighbors[instance] = {}
        # print(f"Processing instance {instance} with {len(cords)} pixels...")
        
        for cord in cords:
            cord_neighbours = search_cord(cord, instance)
            for cord_neighbour,dist in cord_neighbours.items():
                if neighbors[instance].get(cord_neighbour,9999999999) > dist:
                    neighbors[instance][cord_neighbour] = dist

    # Print results and clean up duplicates
    if len(instance_ids) > 3:
        neighbors = prune_neighbors(neighbors_dict=neighbors)
    final_neighbors = {}
    for key, neighbor_distances in neighbors.items():
        # Extract just the neighbor IDs (keys) from the distance dictionary
        neighbor_ids = list(neighbor_distances.keys())
        unique_neighbours = [int(instance_id) for instance_id in neighbor_ids]  # Convert to int and remove duplicates

        # print(f"Instance: {instance_id_to_rgb(key)} neighbors: {unique_neighbours}")
        final_neighbors[int(key)] = unique_neighbours  # Convert key to int for JSON compatibility

    return final_neighbors


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="/netscratch/muhammad/ProcessedDatasets/Fluo-N3DH-SIM+/")
    
    return parser.parse_args()

def main():
    args = parse_args()

    for split in ["val"]:
        gt_paths = sorted(glob.glob(os.path.join(args.root_dir, "gt", split,"*.png")))
        # gt_paths = sorted(glob.glob("C:\\Users\\hasee\\Desktop\\DFKI\\Visual Results\\gt_first_10_test_images\\*.png"))

        neighbors = {}
        for gt_path in tqdm(gt_paths):
            gt = np.array(Image.open(gt_path))
            neighbors_dict = get_neighbors_with_search(gt=gt)
            
            neighbors[gt_path] = neighbors_dict
            
        with open(os.path.join(args.root_dir, "gt", f"{split}_neighbors.json"), "w") as f:
            json.dump(neighbors, f, indent=4)

if __name__ == "__main__":
    main()

