# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 


import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
class InstanceSegmentationLoss(nn.Module):
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
        super(InstanceSegmentationLoss, self).__init__()
        self.name = "InstanceSegmentationLoss"
        self.updated_intra_instance = 0
        self.updated_inter_instance=0
        # self.current_white_loss = 0

    def forward(self, prediction, target, no_bg):
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

            # print(f"{pred_i.shape=}") 640, 648, 3
            # print(f"{gt_i_permute.shape=}") 640, 648, 3

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
                    bg_pred = instance_pred
                    if not no_bg[batch_idx]:
                        # print("background for loss caluclating")
                        # Force background pixels to be near (0,0,0)
                        var_loss = nn.functional.huber_loss(
                            instance_pred,
                            torch.zeros_like(instance_pred)
                        )
                        loss += var_loss
                        ct   += 1
                        instance_means.append(torch.tensor([0, 0, 0]).cuda())
                    else:
                        continue
                else:
                    # Intra-instance variance: push instance_pred toward mean_inst
                    var_loss = nn.functional.huber_loss(
                        instance_pred,
                        torch.full_like(instance_pred, 255)
                    )
                    loss += var_loss
                    ct   += 1

                    # Keep track of the mean for further mean-level separation
                    instance_means.append(mean_inst)
                
                self.updated_intra_instance += var_loss

                
                # ------ Moving Towards semantic segmentation by considereing binary division of classes (background and not background)
                #----Updated inter instance separation------
                size  = instance_mask.sum()  # number of pixels in this instance
                w = 10.0 / torch.sqrt(size.float())
                lambda_sep = 300
                if is_background:
                    # use exactly the above inter-instance separation code except the is_background check
                    non_instance_pred = pred_i[~instance_mask]  # shape: [num_pixels_not_in_instance, 3]
                    if non_instance_pred.numel() > 0:
                        distances = (non_instance_pred - mean_inst).pow(2).sum(dim=1)
                        separation = torch.mean(lambda_sep / (1.0 + distances))
                        loss += w * separation
                else:
                    non_instance_pred = bg_pred
                    if non_instance_pred.numel() > 0:
                        distances = (non_instance_pred - mean_inst).pow(2).sum(dim=1)
                        separation = torch.mean(lambda_sep / (1.0 + distances))
                        loss += w * separation
                self.updated_inter_instance += (w*separation)    

            if ct == 0:
                ct = 1

            total_loss += loss / float(ct)

            self.updated_intra_instance = self.updated_intra_instance / float(ct) / float(batch_size)  
            self.updated_inter_instance = self.updated_inter_instance / float(ct) / float(batch_size)
            
        # Average across batch
        return total_loss / float(batch_size)