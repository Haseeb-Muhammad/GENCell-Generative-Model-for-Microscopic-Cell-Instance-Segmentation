# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class NeighboringLoss(nn.Module):
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
        super(NeighboringLoss, self).__init__()
        self.name = "NeighboringLoss"
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

    def forward(self, prediction, target, no_bg, neighbors):
        """
        Forward pass to compute the custom segmentation loss.
        neighbors: can be either a single dict (old behavior) or a list of dicts (one per batch item)
        """
        # Ensure predictions and targets are float tensors
        prediction = prediction.float()
        target     = target.float()

        batch_size, channels, height, width = prediction.shape
        total_loss = 0.0

        # Handle neighbors as either a single dict or list of dicts
        if isinstance(neighbors, list):
            neighbors_list = neighbors
        else:
            # Old behavior: same neighbors dict for all batch items
            neighbors_list = [neighbors] * batch_size

        # We iterate over the batch dimension explicitly
        for batch_idx in range(batch_size):
            # Get the neighbors dict for this specific batch item
            current_neighbors = neighbors_list[batch_idx]
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

                    # Keep track of the mean for further mean-level separation
                    instance_means.append(mean_inst)

                # ------ Inter-instance separation from neighboring pixels only ------
                current_instance_id = self.rgb_to_instance_id(inst_id.cpu().numpy())
                
                # Get non-instance pixels first
                non_instance_pred = pred_i[~instance_mask]  # shape: [num_pixels_not_in_instance, 3]
                non_instance_gt = gt_i_permute[~instance_mask]  # corresponding GT values [num_pixels_not_in_instance, 3]
                
                # Structured array approach: convert GT RGB to instance IDs vectorized
                if non_instance_pred.numel() > 0 and not is_background:
                    # Check if this instance has neighbors defined
                    if str(current_instance_id) not in current_neighbors:
                        continue  # Skip if no neighbors defined for this instance
                    neighbor_ids = set(int(nid) for nid in current_neighbors[str(current_instance_id)])
                    if no_bg[batch_idx]:
                        neighbor_ids.add(0)

                    # Vectorized conversion of non-instance GT pixels to instance IDs
                    non_instance_ids = (non_instance_gt[:, 0] * 256 * 256 + 
                                      non_instance_gt[:, 1] * 256 + 
                                      non_instance_gt[:, 2]).cpu().numpy()
                    
                    # Create boolean mask for neighboring instances using vectorized operations
                    neighbor_mask = torch.tensor([int(iid) in neighbor_ids for iid in non_instance_ids], 
                                               dtype=torch.bool, device=pred_i.device)
                    
                    # Filter to only neighboring pixels
                    non_instance_pred = non_instance_pred[neighbor_mask]

                    size  = instance_mask.sum()  # number of pixels in this instance
                    w     = 10.0 / torch.sqrt(size.float()) #ORIGINALLY 30 FOR SD2
                    lambda_sep = 300
                    # squared L2 distance from each neighboring pixel to this instance's mean
                    distances   = (non_instance_pred - mean_inst).pow(2).sum(dim=1)
                    separation  = torch.mean(lambda_sep / (1.0 + distances))
                    loss       += w * separation
                    self.current_inter_instance += w*separation

            # -------------- Neighboring Mean-Level Loss ---------------
            # We now have a list of means for each non-background instance: instance_means
            # Let's push neighboring instance means away from each other if they are too close.
            if len(instance_means) > 1:
                means = torch.stack(instance_means, dim=0)[:ct, :]  # shape: [num_means, 3]
                
                # Get corresponding instance IDs for each mean
                unique_instances_list = unique_instances.tolist()[:ct]
                instance_ids = [self.rgb_to_instance_id(np.array(inst)) for inst in unique_instances_list]
                
                # Collect neighboring pairs only
                neighbor_pairs = []
                for i, inst_id in enumerate(instance_ids):
                    if str(inst_id) in current_neighbors:
                        neighbor_list = current_neighbors[str(inst_id)]
                        for neighbor_id in neighbor_list:
                            # Find the index of this neighbor in our instance list
                            try:
                                j = instance_ids.index(int(neighbor_id))
                                if i < j:  # Avoid duplicates by only adding i < j pairs
                                    neighbor_pairs.append((i, j))
                            except ValueError:
                                continue  # Neighbor not in current image
                
                # Compute distances only for neighboring pairs
                if neighbor_pairs:
                    i_indices = torch.tensor([pair[0] for pair in neighbor_pairs])
                    j_indices = torch.tensor([pair[1] for pair in neighbor_pairs])
                    
                    # Get pairwise distances for neighboring instances only
                    pairwise_dists = ((means[i_indices] - means[j_indices]).pow(2).sum(dim=1))
                    
                    # Simple reciprocal penalty or any function of distance
                    # penalty[i,j] = alpha / (dist + eps)
                    lambda_mean = 300.0
                    eps = 1
                    penalty = lambda_mean / (pairwise_dists + eps)
                    
                    # Average across neighboring pairs only
                    mean_separation_loss = penalty.mean()
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