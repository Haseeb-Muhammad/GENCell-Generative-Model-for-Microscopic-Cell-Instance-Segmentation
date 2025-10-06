import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2
import re
from pathlib import Path
import json
import glob
import torchvision.transforms.functional as TF

class Fluo_N3DH_SIM_transform():

    def __init__(self, H, W, split):
        self.resize = transforms.Resize((H,W))
        self.resize_nn = transforms.Resize((H,W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.split = split
    
    def __call__(self, rgb_image, inst_image):
        if random.random() > 0.5 and self.split=="train":
                rgb_image = self.horizontal_flip(rgb_image)
                inst_image = self.horizontal_flip(inst_image)
        rgb_image   = self.resize(rgb_image)
        inst_image = self.resize_nn(inst_image)
        
        # to tensor
        rgb_tensor = self.to_tensor(rgb_image)*2.0-1.0
        inst_tensor = self.to_tensor(inst_image)*255
    
        return rgb_tensor, inst_tensor

class cellTrackingChallengeLoader(Dataset):
    def __init__(self, root_dir, split, transform=True, height=480, width=640):
        self.root_dir = root_dir
        self.split = split
        self.image_paths, self.gt_paths = self._loadPaths()
        if transform == True:
            self.transform = Fluo_N3DH_SIM_transform(H=height, W=width, split=self.split)

    
    def _loadPaths(self):
        gt_paths = sorted(glob.glob(os.path.join(self.root_dir, "gt", self.split, "*.png")))
        image_paths = sorted(glob.glob(os.path.join(self.root_dir, "image", self.split, "*.png")))
        return image_paths, gt_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]

        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        if self.transform:
            rgb_tensor, gt_transformed = self.transform(image, gt)

        return {
            "rgb": rgb_tensor,
            "instance": gt_transformed,
            "no_bg": False,
            "img_path":img_path,
            "gt_path":gt_path
        }