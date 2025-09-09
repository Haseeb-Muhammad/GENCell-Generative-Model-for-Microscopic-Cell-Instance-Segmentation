import argparse
import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import label, regionprops
import copy
from sklearn.metrics import jaccard_score

def color_instances_from_sobel(sobel_image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Labels and colors connected instances in a Sobel edge image.
    This function processes a Sobel edge-detected image to identify and label distinct connected regions (instances).
    It applies thresholding, morphological closing, and connected component labeling. The background label is set
    to the most frequent label in the image to ensure consistency.
    Args:                       
        sobel_image (np.ndarray): Input Sobel edge image as a 2D NumPy array.
        threshold (int, optional): Threshold value for binarizing the edge image. Defaults to 50.
    Returns:
        np.ndarray: A 2D array of the same shape as `sobel_image`, where each connected instance is assigned a unique label.
    """

    # Threshold to create binary edge image
    _, binary = cv2.threshold(sobel_image, threshold, 255, cv2.THRESH_BINARY)

    # Invert binary image to prepare for filling (objects = white)
    inverted = cv2.bitwise_not(binary)

    # Morphological closing to fill gaps in edges
    filled = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Label connected components
    labels = label(filled)

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find the label with the maximum count (most repeated)
    most_repeated_label = unique_labels[np.argmax(counts)]

    # Create a copy of labels to modify
    labels_corrected = labels.copy()

    # If the most repeated label is not already 0, swap it with 0
    if most_repeated_label != 0:
        # Find where the current label 0 is (if it exists)
        mask_zero = (labels == 0)
        mask_most_repeated = (labels == most_repeated_label)
        
        # Swap: most repeated label becomes 0, and old 0 becomes the most repeated label
        labels_corrected[mask_most_repeated] = 0
        # labels_corrected[mask_zero] = most_repeated_label

    return labels_corrected

#@print_param_shapes
def plot_sobel_gradients(slice):
    """
    Computes the combined Sobel gradient magnitude for each color channel of an RGB image slice.
    This function calculates the Sobel gradients in both the x and y directions for each of the
    three color channels (Red, Green, Blue) of the input image. It then computes the gradient
    magnitude for each channel and combines them to produce a single gradient magnitude image.
    Args:
        slice (np.ndarray): An RGB image slice as a NumPy array of shape (H, W, 3).
    Returns:
        np.ndarray: A 2D array representing the combined gradient magnitude of the input image.
    """

    image = cv2.cvtColor(slice, cv2.COLOR_RGB2BGR)
    image = slice
    gradients = {}

    for i, color in enumerate(['B', 'G', 'R']):
        channel = image[:, :, i]
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradients[color] = magnitude

    total_gradient = np.sqrt(
        gradients['R']**2 + gradients['G']**2 + gradients['B']**2
    )
    return total_gradient

def remove_noise(mask : np.array, threshold: int = 20) -> np.array:
    '''
        Remvoes noises from the image based on count of the instance

        Args:
            slice (np.array): 2D gray scale image representing a slice in 3D image s
            threshold (int): threshold for count of instance pixels 
    
        Returns:
            np.array : slice with noise removed through count thresholding
    '''
    instance_ids, counts = np.unique(mask, return_counts=True)
    noise_removed = copy.deepcopy(mask)
    instances_removed=0
    for i, instance_id in enumerate(instance_ids):
        if counts[i] < threshold:
            noise_removed = np.where(noise_removed==instance_id, 0, mask)
            instances_removed+=1

    return noise_removed

import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu

def remove_noise_advanced(mask, connectivity=2):
    """
    Remove small noisy instances from an instance segmentation mask
    using Otsu thresholding on log-area distribution.

    Parameters
    ----------
    mask : ndarray (H, W)
        Instance segmentation mask (background = 0, each instance has a unique label).
    connectivity : int
        Connectivity for connected component labeling (1=4-connectivity, 2=8-connectivity).

    Returns
    -------
    cleaned_mask : ndarray (H, W)
        Mask with small instances removed.
    """

    # Relabel to ensure consecutive labels
    labeled = measure.label(mask > 0, connectivity=connectivity)
    props = measure.regionprops(labeled)

    if not props:
        return mask.copy()  # nothing to clean

    # Compute areas
    areas = np.array([p.area for p in props])
    labels = np.array([p.label for p in props])

    # Log-transform areas
    log_areas = np.log(areas)

    # Otsu threshold on log-areas
    thresh = threshold_otsu(log_areas)

    # Keep labels with log(area) >= threshold
    keep_labels = labels[log_areas >= thresh]

    # Build cleaned mask
    cleaned_mask = np.isin(labeled, keep_labels) * mask

    return cleaned_mask

def remove_noise_with_area(mask : np.array, percentage: int = 1) -> np.array:
    instance_ids, counts = np.unique(mask, return_counts=True)
    noise_removed = copy.deepcopy(mask)
    instances_removed=0
    area = mask.shape[0]*mask.shape[1]
    for i, instance_id in enumerate(instance_ids):
        if counts[i] < area*percentage*0.01:
            noise_removed = np.where(noise_removed==instance_id, 0, mask)
            instances_removed+=1
    print(f"{instances_removed=}")
    return noise_removed



def extract_mask(img:np.array, id:int) -> np.array:
    """
    Extracts a binary mask from the input image where the pixels equal to the specified id are set to 1, and all others are set to 0.

    Args:
        img (np.array): Input image as a NumPy array.
        id (int): The pixel value to extract as a mask.

    Returns:
        np.array: A binary mask of the same shape as `img`, with 1 where `img` equals `id`, and 0 elsewhere.
    """
    return np.where(img==id, 1,0)

def load_tiff(path):
    """Load a 3D TIFF file into a NumPy array."""
    return tiff.imread(path)

def iou_3d(mask1, mask2):
    """Compute 3D IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def evaluate_instance_segmentation(gt_path, pred_path, iou_threshold=0.5):
    # Load masks
    gt = load_tiff(gt_path)
    pred = load_tiff(pred_path)

    # Unique instance IDs (excluding background=0)
    gt_instances = np.unique(gt)
    gt_instances = gt_instances[gt_instances != 0]

    pred_instances = np.unique(pred)
    pred_instances = pred_instances[pred_instances != 0]

    tp, fp, fn = 0, 0, 0
    matched_pred = set()

    for gt_id in gt_instances:
        gt_mask = (gt == gt_id)
        best_iou = 0
        best_pred_id = None

        for pred_id in pred_instances:
            if pred_id in matched_pred:
                continue
            pred_mask = (pred == pred_id)
            iou = iou_3d(gt_mask, pred_mask)
            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou >= iou_threshold:
            tp += 1
            matched_pred.add(best_pred_id)
        else:
            fn += 1

    # Any prediction that wasn't matched to a GT is a false positive
    fp = len(pred_instances) - len(matched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# Example usage:


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--destination_directory",
        type=str,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\codes\\postProcessing\\test.tif"
    )

    parser.add_argument(
        "-s",
        "--source_directory",
        type=str,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\Visual Results\\sobel vs contours vs threshold\\man_seg131_predictioni_vanila_model.tif"
    )

    args = parser.parse_args()
    return args

def main():
    results = evaluate_instance_segmentation(gt_path="/netscratch/muhammad/datasets/Fluo-N3DH-SIM+/Combined/01_GT/SEG/man_seg131.tif", pred_path="/netscratch/muhammad/datasets/Fluo-N3DH-SIM+/Combined/01_RES/mask131.tif", iou_threshold=0.5)
    print(results)

if "__main__" == __name__:
    main()