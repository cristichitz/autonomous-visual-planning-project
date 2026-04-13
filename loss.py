import torch
import torch.nn.functional as F

def compute_semantic_loss_topk(pred_logits, gt_labels, ignore_index=255, top_k_percent=0.2):
    # Per-pixel CE without reduction
    per_pixel_loss = F.cross_entropy(
        pred_logits, gt_labels, ignore_index=ignore_index, reduction='none'
    )  # [B, H, W]
    
    B = per_pixel_loss.shape[0]
    per_pixel_loss = per_pixel_loss.view(B, -1)  # [B, H*W]
    
    num_pixels = per_pixel_loss.shape[1]
    top_k = max(1, int(round(top_k_percent * num_pixels)))
    
    topk_loss, _ = torch.topk(per_pixel_loss, top_k, dim=1)
    
    # Mean over non-zero only
    non_zero = (topk_loss > 0).float().sum(dim=1)
    loss_sum = topk_loss.sum(dim=1)
    per_sample = loss_sum / non_zero.clamp(min=1)
    
    return per_sample.mean()

def compute_loss(predictions, targets, offset_weights, center_weights):
    # Semantic loss (top-k cross entropy)
    sem_loss = compute_semantic_loss_topk(
        predictions['semantic_logits'],
        targets['semantic_masks'],
        ignore_index=255,
        top_k_percent=0.2
    )

    # Instance center heatmap loss (mean squared error)
    center_diff_sq = (predictions['center_heatmap'] - targets['center_heatmaps']) ** 2
    weighted_center_loss = center_diff_sq * center_weights
    num_valid_center = center_weights.sum().clamp(min=1)
    center_loss = weighted_center_loss.sum() / num_valid_center

    # Instance regression loss (L1)
    inst_reg_diff = torch.abs(predictions['center_offsets'] - targets['center_offsets'])
    inst_reg_diff = inst_reg_diff.mean(dim=1, keepdim=True)
    weighted_inst_loss = inst_reg_diff * offset_weights
    num_valid_offset = offset_weights.sum().clamp(min=1)
    inst_reg_loss = weighted_inst_loss.sum() / num_valid_offset

    # Motion Regression Loss (L1)
    motion_reg_diff = torch.abs(predictions['motion_offsets'] - targets['motion_offsets'])
    motion_reg_diff = motion_reg_diff.mean(dim=1, keepdim=True)
    weighted_motion_loss = motion_reg_diff * offset_weights
    motion_reg_loss = weighted_motion_loss.sum() / num_valid_offset

    total_loss = sem_loss + (200.0 * center_loss) + (0.01 * inst_reg_loss) + (0.01 * motion_reg_loss)
    return total_loss, sem_loss, center_loss, inst_reg_loss, motion_reg_loss

def generate_panoptic_targets(instance_masks, sigma=8.0):
    """
    Converts raw instance ID masks into center heatmaps and regression offsets.
    """
    B, H, W = instance_masks.shape
    device = instance_masks.device

    center_heatmaps = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    center_offsets = torch.zeros((B, 2, H, W), device=device, dtype=torch.float32)
    offset_weights = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

    y_coord, x_coord = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )

    for b in range(B):
        unique_ids = torch.unique(instance_masks[b])

        for uid in unique_ids:
            if uid == 0 or uid == 255:
                continue
            mask = (instance_masks[b] == uid)

            y_pixels = y_coord[mask]
            x_pixels = x_coord[mask]
            center_y = y_pixels.mean()
            center_x = x_pixels.mean()

            dist_sq = (y_coord - center_y)**2 + (x_coord - center_x)**2
            gaussian = torch.exp(-dist_sq / (2 * sigma**2))
            center_heatmaps[b, 0] = torch.maximum(center_heatmaps[b, 0], gaussian)
            
            center_offsets[b, 0, mask] = center_y - y_pixels  # Y offset
            center_offsets[b, 1, mask] = center_x - x_pixels  # X offset
            
            offset_weights[b, 0, mask] = 1.0
    return center_heatmaps, center_offsets, offset_weights
