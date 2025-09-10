import torch 
import torch.nn as nn
import numpy as np 

class BinaryCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = "Binary Cross Entropy Loss"
    
    def forward(self, prediction,target):
        prediction = prediction.float()
        target = target.float()

        batch_size, channels, height, width = prediction.shape
        total_loss = 0

        for batch_idx in range(batch_size):
            loss = 0
            
            pred_i = prediction[batch_idx].permute(1,2,0) # [H,W,3]
            gt_i = target[batch_idx].permute(1,2,0) # [H,W,3]

            # print(f"{pred_i.shape=}")
            # print(f"{gt_i.shape=}")

            #converting rgb mask into 1 or 0 gt
            binary_gt_mask = (gt_i!=0).any(dim=-1) # [H,W]
            binary_gt = binary_gt_mask.unsqueeze(0).unsqueeze(0) # [1,H,W]
            # binary_gt = binary_gt_mask.unsqueeze(0) # [1,H,W]
            binary_gt = binary_gt.float()

            # Converting prediction to [0,0,0] or [1,1,1]
            binary_pred_mask = (pred_i>1).any(dim=-1) # [H,W]
            binary_pred = binary_pred_mask.unsqueeze(0).unsqueeze(0) # [1,H,W]
            # binary_pred = binary_pred_mask.unsqueeze(0) # [1,1,H,W]
            binary_pred = binary_pred.float()

            # print(f"{binary_gt.shape=}")
            # print(f"{binary_pred.shape=}")


            
            loss = nn.functional.binary_cross_entropy(
                input=binary_pred,
                target=binary_gt
            )

            total_loss += loss
        
        return total_loss / float(batch_size) 







