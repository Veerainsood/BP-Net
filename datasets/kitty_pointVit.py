import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .dataloaders.my_loader import MyLoader
from .dataloaders.calibration_kitti import Calibration

MEAN = np.array([90.9950, 96.2278, 94.3213], dtype=np.float32)[:,None,None]
STD  = np.array([79.2382, 80.5267, 82.1483], dtype=np.float32)[:,None,None]

class BP2KITTI_Test(Dataset):
    def __init__(self, root_path , height=256, width=1216,*args, **kwargs):
        self.loader = MyLoader(root_path)
        self.height = height
        self.width  = width
    
    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self, idx):
        # 1) raw load
        rgb_np, depth_np, k_cam = self.loader[idx]    # rgb_np: H0×W0×3, depth_np: H0×W0

        # 2) to‐tensor and reorder
        rgb_t    = torch.from_numpy(rgb_np.astype(np.float32)).permute(2,0,1)  # 3×H0×W0
        sparse_t = torch.from_numpy(depth_np.astype(np.float32))[None,:,:]     # 1×H0×W0

        # 3) compute crop offsets
        H0, W0 = rgb_t.shape[1], rgb_t.shape[2]
        tp = H0 - self.height          # how many rows to chop off from top
        lp = (W0 - self.width) // 2    # how many cols to chop off equally on both sides

        # 4) slice out the bottom self.height rows & centered self.width cols
        rgb_t    = rgb_t[:,    tp:tp+self.height,   lp:lp+self.width]
        sparse_t = sparse_t[:, tp:tp+self.height,   lp:lp+self.width]

        # 5) adjust principal point in K_cam
        k_cam[0, 2] -= lp
        k_cam[1, 2] -= tp

        # 6) mirror VirConv’s API: sparse as its own ground truth
        depth_gt = sparse_t.clone()

        return rgb_t, sparse_t, k_cam, depth_gt
