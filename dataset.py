# dataset.py
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

class KittiStepDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=(384, 1248)):
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
        print("Samples test: ", self.samples[0])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_id = sample['sequence_id']
        frame_name = sample['frame_name']

        frame_idx = int(frame_name.replace('.png', ''))
        # Ex: images/train/0000/000000.png
        curr_img_path = os.path.join(self.img_dir, seq_id, frame_name)
        curr_panoptic_path = os.path.join(self.panoptic_dir, seq_id, frame_name)

        prev_frame_name = f"{(frame_idx - 1):06d}.png"
        prev_img_path = os.path.join(self.img_dir, seq_id, prev_frame_name)
        prev_panoptic_path = os.path.join(self.panoptic_dir, seq_id, prev_frame_name)
        
        if not os.path.exists(prev_img_path):
            prev_img_path = curr_img_path
            prev_panoptic_path = curr_panoptic_path

        curr_img = Image.open(curr_img_path).convert('RGB')
        prev_img = Image.open(prev_img_path).convert('RGB')
        
        fixed_size = self.image_size

        # Resize inputs
        curr_img = TF.resize(curr_img, fixed_size)
        prev_img = TF.resize(prev_img, fixed_size)

        curr_panoptic_map = Image.open(curr_panoptic_path)
        curr_panoptic_map = TF.resize(curr_panoptic_map, fixed_size, interpolation=TF.InterpolationMode.NEAREST)

        prev_panoptic_map = Image.open(prev_panoptic_path)
        prev_panoptic_map = TF.resize(prev_panoptic_map, fixed_size, interpolation=TF.InterpolationMode.NEAREST)
        
        curr_tensor = self.normalize(TF.to_tensor(curr_img))
        prev_tensor = self.normalize(TF.to_tensor(prev_img))
        stacked_images = torch.cat([curr_tensor, prev_tensor], dim=0)

        panoptic_np = np.array(curr_panoptic_map, dtype=np.int32)
        semantic_map = panoptic_np[:, :, 0]
        instance_map = panoptic_np[:, :, 1] * 256 + panoptic_np[:, :, 2]

        # Process previous panoptic map
        prev_panoptic_np = np.array(prev_panoptic_map, dtype=np.int32)
        prev_instance_map = prev_panoptic_np[:, :, 1] * 256 + prev_panoptic_np[:, :, 2]
        
        semantic_tensor = torch.from_numpy(semantic_map).long()
        instance_tensor = torch.from_numpy(instance_map).long()
        prev_instance_tensor = torch.from_numpy(prev_instance_map).long()

        return stacked_images, semantic_tensor, instance_tensor, prev_instance_tensor
