# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 


import torch
import torch.nn as nn
import numpy as np
class DistanceLoss(nn.Module):
    """
    Custom Segmentation Loss combining:
      - Intra-instance variance (Huber loss toward the mean for each instance).
      - Inter-instance separation (penalizing closeness of pixels from other instances).
      - Mean-level loss (repulsion among means of different instances).

    Excludes background-labeled pixels (0, 0, 0) unless no_bg[batch] is False.

    Args:
        None    

    Inputs:
        prediction (torch.Tensor): [B, 3, H, W] in [0, 255].
        target (torch.Tensor):     [B, 3, H, W] in [0, 255].
        no_bg (torch.Tensor):      Boolean array of shape [B], where True indicates that
                                   background should not be included for that batch.

    Outputs:
        torch.Tensor: Scalar loss.
    """

    def __init__(self):
        super(DistanceLoss, self).__init__()
        self.name = "InstanceSegmentationLoss"
        self.current_intra_loss = 0
        self.current_inter_instance=0
        self.current_mean_loss = 0

    def rgb_to_instance_id(self, rgb_pixel):
        """Convert RGB pixel to unique instance ID"""
        if np.all(rgb_pixel == 0):  # Background is black (0,0,0)
            return 0
        # Convert RGB to a unique integer ID
        return int(rgb_pixel[0]) * 256 * 256 + int(rgb_pixel[1]) * 256 + int(rgb_pixel[2])

    def instance_id_to_rgb(self, instance_id):
        """Convert instance ID back to RGB"""
        if instance_id == 0:
            return np.array([0, 0, 0])
        r = instance_id // (256 * 256)
        g = (instance_id % (256 * 256)) // 256
        b = instance_id % 256
        return np.array([r, g, b])

    def rgb_tensor_to_instance_id(self, rgb_tensor):
        """Convert RGB tensor to unique instance ID for dictionary lookup"""
        if torch.sum(rgb_tensor).abs() < 1e-5:  # Background is black (0,0,0)
            return 0
        # Convert RGB tensor to a unique integer ID
        return int(rgb_tensor[0].item()) * 256 * 256 + int(rgb_tensor[1].item()) * 256 + int(rgb_tensor[2].item())
    
    def get_distance_weight(self, distances_dict, inst_id1, inst_id2):
        """Get distance-based weight between two instances"""
        if distances_dict is None:
            return 1.0  # Default weight if no distances provided
        
        # Try to get the weight from the distances dictionary
        if inst_id1 in distances_dict and inst_id2 in distances_dict[inst_id1]:
            return distances_dict[inst_id1][inst_id2]
        elif inst_id2 in distances_dict and inst_id1 in distances_dict[inst_id2]:
            return distances_dict[inst_id2][inst_id1]  # Symmetric lookup
        else:
            return 1.0  # Default weight if instances not found in distances

    def forward(self, prediction, target, no_bg, distances):
        """
        Forward pass to compute the custom segmentation loss.

        """
        # Ensure predictions and targets are float tensors
        prediction = prediction.float()
        target     = target.float()

        batch_size, channels, height, width = prediction.shape
        total_loss = 0.0

        # We iterate over the batch dimension explicitly
        for batch_idx in range(batch_size):
            loss = 0.0
            ct   = 0   # count how many instance-related terms contributed

            # Permute predicted channels to [H, W, 3] for easier indexing
            pred_i = prediction[batch_idx].permute(1, 2, 0)  # [H, W, 3]
            gt_i   = target[batch_idx]                       # [3, H, W]
            gt_i_permute = gt_i.permute(1, 2, 0)             # [H, W, 3]

            # Flatten ground-truth instance map to get unique colors (instances)
            gt_i_flat = gt_i_permute.reshape(-1, 3)          # [H*W, 3]
            unique_instances = torch.unique(gt_i_flat, dim=0)# [num_unique, 3]

            # Keep track of means for the non-background instances
            instance_means = []
            instance_ids = []  # Track instance IDs corresponding to means

            # ---------- Main loop over unique instances ----------
            for inst_id in unique_instances:
                if ct > 1250:
                    continue
                # Create a boolean mask for the current instance
                instance_mask = (
                    (gt_i[0] == inst_id[0]) &
                    (gt_i[1] == inst_id[1]) &
                    (gt_i[2] == inst_id[2])
                )

                # Extract the predicted values for pixels belonging to this instance
                instance_pred = pred_i[instance_mask]  # shape: [num_pixels_in_instance, 3]
                if instance_pred.numel() == 0:
                    continue  # Skip if no pixels belong to this instance

                # Compute the mean prediction for this instance
                mean_inst = instance_pred.mean(dim=0)  # shape: [3]
                
                # Check if this is background (sum of inst_id near 0)
                is_background = (torch.sum(inst_id).abs() < 1e-5)

                # ------ Handle background ------
                # Virtual Kitti 2 is not fully annotated (only cars are labeled, so the "background" can contain objects such as trees or traffic lights)
                # Thus, we don't compute loss on the background mask if the sample comes from Virtual Kitti 2
                if is_background:
                    # If background is NOT to be ignored for this batch
                    if not no_bg[batch_idx]:
                        # Force background pixels to be near (0,0,0)
                        var_loss = nn.functional.huber_loss(
                            instance_pred,
                            torch.zeros_like(instance_pred)
                        )
                        loss += var_loss
                        ct   += 1
                        instance_means.append(torch.tensor([0, 0, 0]).cuda())
                        instance_ids.append(0)  # Background instance ID
                        self.current_intra_loss += var_loss

                    else:
                        # If background is ignored, skip
                        continue
                else:
                    # Intra-instance variance: push instance_pred toward mean_inst
                    var_loss = nn.functional.huber_loss(
                        instance_pred,
                        mean_inst.unsqueeze(0).expand_as(instance_pred)
                    )
                    loss += var_loss
                    ct   += 1
                    self.current_intra_loss += var_loss

                    # Keep track of the mean and instance ID for further mean-level separation
                    instance_means.append(mean_inst)
                    instance_ids.append(self.rgb_tensor_to_instance_id(inst_id))

                # ------ Inter-instance separation from other pixels ------
                if not is_background:
                    current_inst_id = self.rgb_tensor_to_instance_id(inst_id)
                    
                    # Vectorized approach: process all other instances at once
                    # Create mask for all pixels NOT belonging to current instance
                    not_current_mask = ~instance_mask
                    other_pixels = pred_i[not_current_mask]  # [num_other_pixels, 3]
                    
                    if other_pixels.numel() > 0:
                        # Get the ground truth labels for these other pixels
                        other_gt_pixels = gt_i_permute[not_current_mask]  # [num_other_pixels, 3]
                        
                        # Vectorized weight calculation using unique instances and broadcasting
                        # Get unique instance IDs from other pixels
                        unique_other_instances = torch.unique(other_gt_pixels, dim=0)
                        
                        # Create a mask and weight tensor for all other pixels
                        weights = torch.ones(other_pixels.shape[0], device=prediction.device)
                        
                        # Process each unique other instance
                        for other_inst_rgb in unique_other_instances:
                            other_inst_id = self.rgb_tensor_to_instance_id(other_inst_rgb)
                            weight = self.get_distance_weight(distances, current_inst_id, other_inst_id)
                            
                            # Create mask for pixels belonging to this other instance
                            other_inst_mask = (
                                (other_gt_pixels[:, 0] == other_inst_rgb[0]) &
                                (other_gt_pixels[:, 1] == other_inst_rgb[1]) &
                                (other_gt_pixels[:, 2] == other_inst_rgb[2])
                            )
                            weights[other_inst_mask] = weight
                        
                        size = instance_mask.sum()  # number of pixels in current instance
                        w_base = 10.0 / torch.sqrt(size.float())
                        lambda_sep = 300
                        
                        # Vectorized distance calculation: [num_other_pixels, 3] - [3] -> [num_other_pixels, 3]
                        pixel_distances = (other_pixels - mean_inst).pow(2).sum(dim=1)  # [num_other_pixels]
                        separations = lambda_sep / (1.0 + pixel_distances)  # [num_other_pixels]
                        
                        # Apply weights and compute weighted mean
                        weighted_separations = separations * weights  # [num_other_pixels]
                        total_separation = torch.mean(weighted_separations) * w_base
                        
                        loss += total_separation
                        self.current_inter_instance += total_separation

            # -------------- Vectorized Mean-Level Loss with Distance-based Weights ---------------
            # We now have a list of means for each non-background instance: instance_means
            # And corresponding instance IDs: instance_ids
            # Let's push them away from each other using distance-based weights.

            if len(instance_means) > 1:
                means = torch.stack(instance_means, dim=0)[:ct, :]  # shape: [num_means, 3]
                
                # Compute the pairwise squared distances in a vectorized manner:
                # differences: [num_means, num_means, 3]
                differences = means.unsqueeze(1) - means.unsqueeze(0)
                # squared_distances: [num_means, num_means]
                squared_distances = differences.pow(2).sum(dim=2)

                # Create weight matrix for all pairs
                weight_matrix = torch.ones_like(squared_distances)
                
                # Fill weight matrix using distance dictionary - vectorized approach
                if distances is not None:
                    num_means = min(ct, len(instance_ids))
                    # Create tensor of instance IDs for vectorized lookup
                    ids_tensor = torch.tensor(instance_ids[:num_means], device=prediction.device)
                    
                    # For each pair, look up the weight
                    for i in range(num_means):
                        for j in range(num_means):
                            if i != j:
                                weight = self.get_distance_weight(distances, instance_ids[i], instance_ids[j])
                                weight_matrix[i, j] = weight

                # We only want i < j to avoid double-counting or i=j
                # shape: [2, #pairs]
                i_indices, j_indices = torch.triu_indices(
                    squared_distances.size(0),
                    squared_distances.size(1),
                    offset=1
                )
                
                # Get pairwise distances and weights for upper triangular part
                pairwise_dists = squared_distances[i_indices, j_indices]
                pairwise_weights = weight_matrix[i_indices, j_indices]

                # Vectorized penalty calculation
                lambda_mean = 300.0
                eps = 1
                penalties = pairwise_weights * lambda_mean / (pairwise_dists + eps)
                
                # Average across all weighted pairs
                mean_separation_loss = torch.mean(penalties)
                loss += mean_separation_loss
                self.current_mean_loss += mean_separation_loss

            # Avoid dividing by zero if ct was never incremented
            if ct == 0:
                ct = 1

            total_loss += loss / float(ct)
            self.current_intra_loss = (self.current_intra_loss / float(ct)) / float(batch_size)
            self.current_inter_instance = (self.current_inter_instance / float(ct)) / float(batch_size)
            self.current_mean_loss = (self.current_mean_loss / float(ct)) / float(batch_size)
            
        # Average across batch
        return total_loss / float(batch_size)