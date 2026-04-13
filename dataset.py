# dataset.py
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


class JointPreprocessor:
    def __init__(self, crop_size=(385, 1249), is_training=True, ignore_label=255):
        self.crop_size = crop_size
        self.is_training = is_training
        self.ignore_label = ignore_label
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __call__(self, curr_img, prev_img, curr_mask, prev_mask):
        if self.is_training:
            # 1. Random Scale (Discrete steps from 0.5 to 2.0, step 0.1)
            scale_factor = random.choice([x / 10.0 for x in range(5, 21)])
            new_h, new_w = int(curr_img.height * scale_factor), int(curr_img.width * scale_factor)
            
            curr_img = TF.resize(curr_img, (new_h, new_w))
            prev_img = TF.resize(prev_img, (new_h, new_w))
            curr_mask = TF.resize(curr_mask, (new_h, new_w), TF.InterpolationMode.NEAREST)
            prev_mask = TF.resize(prev_mask, (new_h, new_w), TF.InterpolationMode.NEAREST)

            # 2. Pad if smaller than crop size (Using ignore_label for masks!)
            pad_h, pad_w = max(self.crop_size[0] - new_h, 0), max(self.crop_size[1] - new_w, 0)
            if pad_h > 0 or pad_w > 0:
                curr_img = TF.pad(curr_img, (0, 0, pad_w, pad_h))
                prev_img = TF.pad(prev_img, (0, 0, pad_w, pad_h))
                curr_mask = TF.pad(curr_mask, (0, 0, pad_w, pad_h), fill=self.ignore_label)
                prev_mask = TF.pad(prev_mask, (0, 0, pad_w, pad_h), fill=self.ignore_label)

            # 3. Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(curr_img, output_size=self.crop_size)
            curr_img = TF.crop(curr_img, i, j, h, w)
            prev_img = TF.crop(prev_img, i, j, h, w)
            curr_mask = TF.crop(curr_mask, i, j, h, w)
            prev_mask = TF.crop(prev_mask, i, j, h, w)

            # 4. Random Horizontal Flip
            if random.random() > 0.5:
                curr_img, prev_img = TF.hflip(curr_img), TF.hflip(prev_img)
                curr_mask, prev_mask = TF.hflip(curr_mask), TF.hflip(prev_mask)

            # 5. Photometric Transform (Images ONLY)
            curr_img, prev_img = self.color_jitter(curr_img), self.color_jitter(prev_img)
            
        else:
            curr_img = TF.resize(curr_img, self.crop_size)
            prev_img = TF.resize(prev_img, self.crop_size)
            curr_mask = TF.resize(curr_mask, self.crop_size, TF.InterpolationMode.NEAREST)
            prev_mask = TF.resize(prev_mask, self.crop_size, TF.InterpolationMode.NEAREST)

        # Convert to Normalized Tensors
        curr_tensor = self.normalize(TF.to_tensor(curr_img))
        prev_tensor = self.normalize(TF.to_tensor(prev_img))
        stacked_images = torch.cat([curr_tensor, prev_tensor], dim=0)

        return stacked_images, curr_mask, prev_mask

class TargetGenerator:
    def __init__(self, thing_ids=(11, 13), ignore_label=255, sigma=8.0):
        self.thing_ids = thing_ids
        self.ignore_label = ignore_label
        self.sigma = sigma

    def extract(self, mask_pil):
        mask_np = np.array(mask_pil, dtype=np.int32)
        semantic_map = mask_np[:, :, 0]
        instance_map = mask_np[:, :, 1] * 256 + mask_np[:, :, 2]

        instance_map[semantic_map == self.ignore_label] = 0
        is_thing = np.isin(semantic_map, self.thing_ids)
        instance_map[~is_thing] = 0

        is_crowd = is_thing & (instance_map == 0)
        instance_map[is_crowd] = 0
        
        return torch.from_numpy(semantic_map).long(), torch.from_numpy(instance_map).long()

class KittiStepDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=(385, 1249)):
        """
        Args:
            root_dir: Path to KITTI_STEP_ROOT
            split: 'train' or 'val'
            image_size: (heigh, width) to resize inputs to.
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.img_dir = os.path.join(root_dir, 'images', split)
        self.panoptic_dir = os.path.join(root_dir, 'panoptic_maps', split)

        # Initialize the external engines
        self.preprocessor = JointPreprocessor(crop_size=image_size, is_training=(split == 'train'))
        self.extractor = TargetGenerator(thing_ids=(11, 13), ignore_label=255)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.samples = []
        for sequence_id in sorted(os.listdir(self.img_dir)):
            seq_path = os.path.join(self.img_dir, sequence_id)
            if not os.path.isdir(seq_path): continue

            frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame_name in frames:
                self.samples.append({
                    'sequence_id': sequence_id,
                    'frame_name': frame_name
                })
        print(f"Loaded {len(self.samples)} {split} samples. Example: ", self.samples[0])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_id = sample['sequence_id']
        frame_name = sample['frame_name']

        frame_idx = int(frame_name.replace('.png', ''))
        # Ex: images/train/0000/000000.png
        curr_img_path = os.path.join(self.img_dir, seq_id, frame_name)

        prev_frame_name = f"{(frame_idx - 1):06d}.png"
        prev_img_path = os.path.join(self.img_dir, seq_id, prev_frame_name)
        
        if not os.path.exists(prev_img_path):
            prev_img_path = curr_img_path
        
        curr_img = Image.open(curr_img_path).convert('RGB')
        prev_img = Image.open(prev_img_path).convert('RGB')


        if self.split == 'test':
            # Test mode: No masks exist. Just resize, normalize, and return.
            curr_img = TF.resize(curr_img, self.image_size)
            prev_img = TF.resize(prev_img, self.image_size)
            curr_tensor = self.normalize(TF.to_tensor(curr_img))
            prev_tensor = self.normalize(TF.to_tensor(prev_img))
            stacked_images = torch.cat([curr_tensor, prev_tensor], dim=0)
            
            return stacked_images
        
        curr_panoptic_path = os.path.join(self.panoptic_dir, seq_id, frame_name)
        prev_panoptic_path = os.path.join(self.panoptic_dir, seq_id, prev_frame_name)

        if not os.path.exists(prev_panoptic_path):
            prev_panoptic_path = curr_panoptic_path

        curr_panoptic_map = Image.open(curr_panoptic_path)
        prev_panoptic_map = Image.open(prev_panoptic_path)

        stacked_images, curr_mask_aug, prev_mask_aug = self.preprocessor(
            curr_img, prev_img, curr_panoptic_map, prev_panoptic_map
        )

        curr_sem, curr_inst = self.extractor.extract(curr_mask_aug)
        prev_sem, prev_inst = self.extractor.extract(prev_mask_aug)

        return stacked_images, curr_sem, curr_inst, prev_inst
