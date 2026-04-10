import  torch
import  os
from    torch.amp import autocast, GradScaler
from    model import MotionDeepLab
from    loss  import compute_loss, generate_panoptic_targets
from    dataset import KittiStepDataset
from    torch.utils.data import DataLoader

class GPUMathTargetGenerator:
    """Performs all heatmap and offset math on the GPU for a full batch."""
    def __init__(self, device, image_size=(384, 1248), ignore_label=255, sigma=8.0):
        self.device = device
        self.ignore_label = ignore_label
        self.sigma = sigma
        
        H, W = image_size
        y_coord, x_coord = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        self.y_coord = y_coord.unsqueeze(0) # Shape: (1, H, W)
        self.x_coord = x_coord.unsqueeze(0)

    def generate(self, curr_inst, prev_inst):
        B, H, W = curr_inst.shape
        
        center_heatmaps = torch.zeros((B, 1, H, W), dtype=torch.float32, device=self.device)
        prev_heatmaps = torch.zeros((B, 1, H, W), dtype=torch.float32, device=self.device)
        center_offsets = torch.zeros((B, 2, H, W), dtype=torch.float32, device=self.device)
        motion_offsets = torch.zeros((B, 2, H, W), dtype=torch.float32, device=self.device)
        offset_weights = torch.zeros((B, 1, H, W), dtype=torch.float32, device=self.device)

        # Process each item in the batch
        for b in range(B):
            c_inst = curr_inst[b]
            p_inst = prev_inst[b]
            
            unique_ids = torch.unique(c_inst)

            for uid in unique_ids:
                if uid == 0 or uid == self.ignore_label:
                    continue
                
                mask = (c_inst == uid)
                y_pixels = self.y_coord[0][mask]
                x_pixels = self.x_coord[0][mask]
                
                center_y, center_x = y_pixels.mean(), x_pixels.mean()

                dist_sq = (self.y_coord[0] - center_y)**2 + (self.x_coord[0] - center_x)**2
                gaussian = torch.exp(-dist_sq / (2 * self.sigma**2))
                center_heatmaps[b, 0] = torch.maximum(center_heatmaps[b, 0], gaussian)
                
                center_offsets[b, 0, mask] = center_y - y_pixels
                center_offsets[b, 1, mask] = center_x - x_pixels
                offset_weights[b, 0, mask] = 1.0

                prev_mask = (p_inst == uid)
                if prev_mask.any():
                    prev_y_pixels = self.y_coord[0][prev_mask]
                    prev_x_pixels = self.x_coord[0][prev_mask]
                    prev_center_y, prev_center_x = prev_y_pixels.mean(), prev_x_pixels.mean()
                    
                    prev_dist_sq = (self.y_coord[0] - prev_center_y)**2 + (self.x_coord[0] - prev_center_x)**2
                    prev_gaussian = torch.exp(-prev_dist_sq / (2 * self.sigma**2))
                    prev_heatmaps[b, 0] = torch.maximum(prev_heatmaps[b, 0], prev_gaussian)
                    
                    motion_offsets[b, 0, mask] = prev_center_y - y_pixels
                    motion_offsets[b, 1, mask] = prev_center_x - x_pixels

        return center_heatmaps, prev_heatmaps, center_offsets, motion_offsets, offset_weights

class  Trainer:
    def __init__(self, root_dir, batch_size=4, epochs=25, accumulation_steps=4, current_model_path=None, save_freq=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.current_model_path = current_model_path
        self.save_freq = save_freq
        self.train_ds = KittiStepDataset(root_dir=root_dir, split='train')
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        self.total_steps = self.epochs * len(self.train_loader)
        self.model = MotionDeepLab(
            thing_class_ids=[11, 13],
            label_divisor=1000,
            void_label=255
        ).to(self.device)

        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if 'encoder' in name: 
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-4} 
        ], weight_decay=1e-4)

        self.scaler = GradScaler()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=self._poly_decay
        )
        
        self.start_epoch = 1
        self._maybe_load_checkpoint()

    def _poly_decay(self, current_step):
        poly_power = 0.9
        if current_step >= self.total_steps:
            return 0.0
        return (1.0 - current_step / self.total_steps) ** poly_power
    
    def _maybe_load_checkpoint(self):
        if self.current_model_path and os.path.exists(self.current_model_path):
            checkpoint = torch.load(self.current_model_path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming full state from Epoch {self.start_epoch}")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                self.start_epoch = 50  # Assuming you already did 50
                print(f"Loaded raw weights. Optimizer starting cold at Epoch {self.start_epoch}.")

    def _train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        running_loss = 0.0

        if not hasattr(self, 'gpu_target_gen'):
            self.gpu_target_gen = GPUMathTargetGenerator(self.device)

        for i, batch in enumerate(self.train_loader):
            images, curr_sem, curr_inst, prev_inst = [b.to(self.device) for b in batch]

            with torch.no_grad():
                heatmaps, prev_heatmaps, center_offsets, motion_offsets, offset_weights = \
                    self.gpu_target_gen.generate(curr_inst, prev_inst)

            targets = {
                'semantic_masks': curr_sem,
                'center_heatmaps': heatmaps,
                'center_offsets': center_offsets,
                'motion_offsets': motion_offsets
            }

            # Forward and Loss Calculation
            with autocast(device_type='cuda'):
                predictions = self.model(images, gt_prev_heatmap=prev_heatmaps)
                total_loss, _, _, _, _ = compute_loss(predictions, targets, offset_weights)
            
            # Synchronize outside of mixed precision context
            loss_value = total_loss.item()
            total_loss = total_loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(total_loss).backward()
            running_loss += loss_value

            # Gradient Accumulation Step
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                current_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if current_scale <= self.scaler.get_scale():
                    self.scheduler.step()
                
                self.optimizer.zero_grad()

            # Logging
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Epoch [{epoch}/{self.epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

    def fit(self):
        print("Starting training!")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)
            
            if epoch % self.save_freq == 0 or epoch == self.epochs:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict()
                }
                save_path = f'weights/motion_deeplab_epoch_{epoch}.pth'
                os.makedirs('weights', exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"Saved to {save_path}")
            print(f"Epoch {epoch} Complete.")


if __name__ == '__main__':
    trainer = Trainer(
        root_dir='.',
        batch_size=4,
        epochs=100,  # e.g., run to 100 total
        accumulation_steps=4,
        current_model_path='weights/motion_deeplab_epoch_49.pth' # Path to your current weights
    )
    trainer.fit()