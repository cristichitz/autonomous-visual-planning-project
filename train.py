import  torch
import  os
from    torch.amp import autocast, GradScaler
from    model import MotionDeepLab
from    loss  import compute_loss, generate_panoptic_targets
from    dataset import KittiStepDataset
from    torch.utils.data import DataLoader


KITTI_THING_IDS = [11, 13]
KITTI_STEP_ROOT = '.'
train_ds = KittiStepDataset(root_dir=KITTI_STEP_ROOT, split='train')

BATCH_SIZE = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

for stacked_images, semantic_masks, instance_masks, prev_inst_masks in train_loader:
    print("Batch loaded succesfully!")
    print(f"Images shape:    {stacked_images.shape}")  
    print(f"Semantics shape: {semantic_masks.shape}")  
    print(f"Instances shape: {instance_masks.shape}") 
    print(f"Prev Instances shape: {prev_inst_masks.shape}")  

    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MotionDeepLab(
    thing_class_ids=KITTI_THING_IDS,
    label_divisor=1000,    # Updated based on your config!
    void_label=255         # Confirmed by ignore_label=255 in your config
).to(device)

backbone_params, head_params = [], []
for name, param in model.named_parameters():
    if 'encoder' in name: 
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-4} 
], weight_decay=1e-4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scaler = GradScaler()
EPOCHS = 25
ACCUMULATION_STEPS = 4
CURRENT_MODEL = 'weights/motion_deeplab_epoch_10.pth'
RESUME_TRAINING = True

total_steps = EPOCHS * len(train_loader)
poly_power = 0.9

def poly_decay(current_step):
    if current_step >= total_steps:
        return 0.0
    return (1.0 - current_step / total_steps) ** poly_power

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_decay) 

start_epoch = 1
if RESUME_TRAINING and os.path.exists(CURRENT_MODEL):
    checkpoint = torch.load(CURRENT_MODEL)
    
    # Check if this is the new dictionary format or the old raw weights
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state']) # Fixed typo here
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming full state from Epoch {start_epoch}")
    else:
        # Fallback for your Epoch 10 raw weights
        model.load_state_dict(checkpoint, strict=False)
        start_epoch = 11  # You know the old model was epoch 10
        print(f"Loaded raw weights from {CURRENT_MODEL}. Optimizer starting cold at Epoch 11.")

print("Starting training!")
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    running_loss = 0
    for i, (images, sem_masks, inst_masks, prev_inst_masks) in enumerate(train_loader):
        images = images.to(device)
        sem_masks = sem_masks.to(device)
        inst_masks = inst_masks.to(device)
        prev_inst_masks = prev_inst_masks.to(device)

        gt_heatmaps, gt_inst_offsets, offset_weights = generate_panoptic_targets(inst_masks)
        
        # We need the motion targets as well (distance from current pixel to Previous frame)
        prev_heatmaps, gt_motion_offsets, _ = generate_panoptic_targets(prev_inst_masks)

        targets = {
            'semantic_masks': sem_masks,
            'center_heatmaps': gt_heatmaps,
            'center_offsets': gt_inst_offsets,
            'motion_offsets': gt_motion_offsets
        }
        # model_input = torch.cat([images, prev_heatmaps], dim=1)

        with autocast(device_type='cuda'):
            predictions = model(images, gt_prev_heatmap=prev_heatmaps)
            total_loss, _, _, _, _ = compute_loss(predictions, targets, offset_weights)
            loss_value = total_loss.item()
            total_loss = total_loss / ACCUMULATION_STEPS

        scaler.scale(total_loss).backward()
        running_loss += loss_value
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            current_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # 2. Only step the scheduler if the scale didn't drop 
            # (A dropped scale means gradients were bad and optimizer was skipped)
            if current_scale <= scaler.get_scale():
                scheduler.step()
            optimizer.zero_grad()
        if (i + 1) % 50 == 0:
            avg_loss = running_loss / 50
            print(f"Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
            running_loss = 0.0

    print(f"Epoch {epoch} Complete.")
    checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict()
    }
    torch.save(checkpoint, f'motion_deeplab_epoch_{epoch}.pth')
    print(f"Model saved to motion_deeplab_epoch_{epoch}.pth")
