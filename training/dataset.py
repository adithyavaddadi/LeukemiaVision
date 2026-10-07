import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

from .preprocess import aspect_safe_pad_to_square
from .stain import StainNormalizer

class CNMCDataset(Dataset):
    """
    Authoritative PyTorch Dataset for C-NMC 2019 single-cell smear images.
    Supports aspect-safe padding, stain normalization, and morphology-preserving augmentations.
    """
    def __init__(self, df, img_size=224, is_train=False, stain_method="none", augment_cfg=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train
        self.stain_normalizer = StainNormalizer(method=stain_method)
        self.augment_cfg = augment_cfg or {}
        
        # Base normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def _augment(self, img_pil):
        # Morphology-preserving augmentations only
        # 1. Random horizontal flip
        if random.random() < self.augment_cfg.get("horizontal_flip_prob", 0.5):
            img_pil = TF.hflip(img_pil)
            
        # 2. Random vertical flip
        if random.random() < self.augment_cfg.get("vertical_flip_prob", 0.5):
            img_pil = TF.vflip(img_pil)
            
        # 3. Random 90/180/270 degree rotation
        if random.random() < self.augment_cfg.get("rotation_90_prob", 0.5):
            angle = random.choice([90, 180, 270])
            img_pil = TF.rotate(img_pil, angle)
            
        # 4. Small rotation (+- 15 degrees)
        rot_limit = self.augment_cfg.get("affine_rotation_limit", 15)
        if rot_limit > 0 and random.random() < 0.5:
            deg = random.uniform(-rot_limit, rot_limit)
            img_pil = TF.rotate(img_pil, deg)
            
        # 5. Mild brightness / contrast jitter
        if random.random() < self.augment_cfg.get("brightness_contrast_prob", 0.3):
            b_factor = 1.0 + random.uniform(-0.1, 0.1)
            c_factor = 1.0 + random.uniform(-0.1, 0.1)
            img_pil = TF.adjust_brightness(img_pil, b_factor)
            img_pil = TF.adjust_contrast(img_pil, c_factor)
            
        return img_pil

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['numeric_label'])
        subject_id = str(row['subject_id'])
        
        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        
        # Apply aspect-safe pad to square
        img_pil = aspect_safe_pad_to_square(img_pil)
        
        # Apply stain normalization if active
        if self.stain_normalizer.method != "none":
            np_arr = np.array(img_pil)
            norm_np = self.stain_normalizer.normalize(np_arr)
            img_pil = Image.fromarray((norm_np * 255).astype(np.uint8))
            
        # Apply augmentation if in training mode
        if self.is_train:
            img_pil = self._augment(img_pil)
            
        # Resize to model resolution
        img_resized = img_pil.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
        
        # Convert to tensor and normalize
        tensor = TF.to_tensor(img_resized)
        tensor = self.normalize(tensor)
        
        return tensor, torch.tensor(label, dtype=torch.long), subject_id, img_path
