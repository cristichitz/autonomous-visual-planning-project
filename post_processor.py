import torch
import torch.nn as nn
import torch.nn.functional as F

class PanopticPostProcessor(nn.Module):
    def __init__(self, thing_class_ids, center_threshold=0.1, nms_kernel=9,
                 keep_k_centers=200, label_divisor=256, stuff_area_limit=4096, void_label=255):
        super().__init__()

        self.register_buffer('thing_class_ids', torch.tensor(thing_class_ids, dtype=torch.long))
        self.center_threshold = center_threshold
        self.nms_kernel = nms_kernel
        self.keep_k_centers = keep_k_centers
        self.label_divisor = label_divisor
        self.stuff_area_limit = stuff_area_limit
        self.void_label = void_label
    
    @torch.no_grad()
    def forward(self, semantic_logits, center_heatmap, offset_map):
        """
        Inputs:
            semantic_logits: [B, Num_Classes, H, W]
            center_heatmap:  [B, 1, H, W]
            offset_map:      [B, 2, H, W] (Channel 0: Y, Channel 1: X)
        """

        B, C, H, W = semantic_logits.shape
        semantic_pred = torch.argmax(semantic_logits, dim=1) # [B, H, W]
        
        panoptic_preds, instance_preds = [], []

        for i in range(B):
            sem = semantic_pred[i]
            heat = center_heatmap[i:i+1]
            offset = offset_map[i]

            heat_thresh = torch.where(heat > self.center_threshold, heat, torch.zeros_like(heat))
            pad = self.nms_kernel // 2
            pooled = F.max_pool2d(heat_thresh, kernel_size=self.nms_kernel, stride=1, padding=pad)

            heat_nms = torch.where(heat_thresh == pooled, heat_thresh, torch.zeros_like(heat_thresh))[0, 0] # [H, W]
            
            centers_y, centers_x = torch.nonzero(heat_nms > 0, as_tuple=True)
            scores = heat_nms[centers_y, centers_x]
            
            if self.keep_k_centers > 0 and len(scores) > self.keep_k_centers:
                topk_scores, topk_idx = torch.topk(scores, self.keep_k_centers)
                centers_y, centers_x = centers_y[topk_idx], centers_x[topk_idx]

            if len(centers_y) == 0:
                instance_map = torch.zeros_like(sem)
            else:
                y_coord, x_coord = torch.meshgrid(
                    torch.arange(H, device=sem.device), 
                    torch.arange(W, device=sem.device), 
                    indexing='ij'
                )
                target_y = y_coord.float() + offset[0]
                target_x = x_coord.float() + offset[1]
                pixel_targets = torch.stack([target_y, target_x], dim=-1).view(-1, 2) # [H*W, 2]
                
                centers = torch.stack([centers_y.float(), centers_x.float()], dim=1)

                distances = torch.cdist(pixel_targets, centers) # [H*W, N]
                closest_center_idx = torch.argmin(distances, dim=1).view(H, W)
                
                # Mask out 'stuff' pixels so they don't get instance IDs
                thing_mask = torch.isin(sem, self.thing_class_ids)
                instance_map = (closest_center_idx + 1) * thing_mask.int() # IDs start at 1

            
            panoptic_map = torch.ones_like(sem) * self.void_label * self.label_divisor
            
            # 3a. Paste 'Thing' instances with majority-vote semantic classes
            for inst_id in torch.unique(instance_map):
                if inst_id == 0: continue
                inst_mask = (instance_map == inst_id)
                sem_classes_in_inst = sem[inst_mask]
                
                if len(sem_classes_in_inst) == 0: continue
                
                majority_class = torch.mode(sem_classes_in_inst).values
                panoptic_map[inst_mask] = (majority_class * self.label_divisor) + inst_id
            
            # 3b. Paste 'Stuff' regions (filtering out small disconnected regions)
            stuff_mask = (instance_map == 0)
            for sem_id in torch.unique(sem):
                if sem_id in self.thing_class_ids: continue
                
                current_stuff_mask = (sem == sem_id) & stuff_mask
                if current_stuff_mask.sum() >= self.stuff_area_limit:
                    panoptic_map[current_stuff_mask] = sem_id * self.label_divisor
                    
            panoptic_preds.append(panoptic_map)
            instance_preds.append(instance_map)
            
        return torch.stack(panoptic_preds), torch.stack(instance_preds)
    

class MotionTracker:
    def __init__(self, label_divisor, void_label, sigma_render=8, sigma_track=7):
        self.label_divisor = label_divisor
        self.void_label = void_label
        self.sigma_render = sigma_render
        self.sigma_track = sigma_track

        self.prev_centers = None
        self.next_tracking_id = 1
    
    def reset_state(self, device):
        """Called when a new video sequence starts."""
        self.prev_centers = torch.empty((0, 5), dtype=torch.int32, device=device)
        self.next_tracking_id = 1
    
    def render_panoptic_map_as_heatmap(self, panoptic_map):
        """Extracts centers from panoptic map and renders a 2D Gaussian heatmap."""
        H, W = panoptic_map.shape
        device = panoptic_map.device

        gaussian_size = 6 * self.sigma_render + 3
        # Create 2D Gaussian patch
        x = torch.arange(gaussian_size, dtype=torch.float32, device=device)
        y = x.unsqueeze(1)
        center_coord = 3 * self.sigma_render + 1
        gaussian = torch.exp(-((x - center_coord)**2 + (y - center_coord)**2) / (2 * self.sigma_render**2))
        pad_begin = int(round(3 * self.sigma_render + 1))
        pad_total = pad_begin + int(round(3 * self.sigma_render + 2))
        
        center_map = torch.zeros((H + pad_total, W + pad_total), device=device)
        centers_and_ids = []

        for pan_id in torch.unique(panoptic_map):
            sem_id = pan_id // self.label_divisor
            if sem_id == self.void_label or pan_id % self.label_divisor == 0:
                continue
        
            mask = (panoptic_map == pan_id)
            coords = torch.nonzero(mask).float() # [N, 2] -> (y, x)

            if len(coords) == 0: continue

            # Radius proxy: width * height of bounding box
            min_y, min_x = coords[:, 0].min(), coords[:, 1].min()
            max_y, max_x = coords[:, 0].max(), coords[:, 1].max()
            radius_sq = int(round(float((max_y - min_y) * (max_x - min_x))))
            
            center_y, center_x = coords.mean(dim=0)
            cy, cx = int(torch.round(center_y)), int(torch.round(center_x))
            
            centers_and_ids.append([cx, cy, int(pan_id), radius_sq, 0])

            y_start, y_end = cy, cy + gaussian_size
            x_start, x_end = cx, cx + gaussian_size
            center_map[y_start:y_end, x_start:x_end] = torch.maximum(
                center_map[y_start:y_end, x_start:x_end], gaussian
            )
            
        center_map = center_map[pad_begin:pad_begin+H, pad_begin:pad_begin+W]
        
        centers_tensor = torch.tensor(centers_and_ids, dtype=torch.int32, device=device) if centers_and_ids else torch.empty((0, 5), dtype=torch.int32, device=device)
            
        return center_map.unsqueeze(0).unsqueeze(0), centers_tensor
    
    def assign_instances_to_previous_tracks(self, current_centers, heatmap, offsets, panoptic_map):
        """Matches current objects to previous frame's objects to maintain track IDs."""
        device = panoptic_map.device
        if self.prev_centers is None:
            self.reset_state(device)
            
        if len(current_centers) == 0:
            return panoptic_map.clone()
            
        heatmap = heatmap.squeeze() # [H, W]
        offsets = offsets.squeeze() # [2, H, W] -> 0 is Y, 1 is X
        
        # Sort centers by confidence
        cy, cx = current_centers[:, 1].long(), current_centers[:, 0].long()
        scores = heatmap[cy, cx]
        current_centers = current_centers[torch.argsort(scores, descending=True)]
        
        new_panoptic_map = panoptic_map.clone()
        updated_current_centers = []
        prev_matched = torch.zeros(len(self.prev_centers), dtype=torch.bool, device=device)
        
        for i in range(len(current_centers)):
            center = current_centers[i].clone()
            cx, cy = center[0].long(), center[1].long()
            center_id = center[2]
            
            # Grab temporal offset vectors
            offset_y = offsets[0, cy, cx]
            offset_x = offsets[1, cy, cx]
            
            proj_x = cx.float() + offset_x
            proj_y = cy.float() + offset_y
            
            sem_id = center_id // self.label_divisor
            center_mask = (panoptic_map == center_id)
            match_found = False
            
            if len(self.prev_centers) > 0:
                prev_classes = self.prev_centers[:, 2] // self.label_divisor
                valid_mask = (prev_classes == sem_id) & (~prev_matched)
                valid_indices = torch.nonzero(valid_mask).squeeze(1)
                
                if len(valid_indices) > 0:
                    valid_prev = self.prev_centers[valid_indices]
                    dist_sq = (valid_prev[:, 0].float() - proj_x)**2 + (valid_prev[:, 1].float() - proj_y)**2
                    
                    min_dist_idx = torch.argmin(dist_sq)
                    min_dist = dist_sq[min_dist_idx]
                    
                    original_prev_idx = valid_indices[min_dist_idx]
                    prev_radius_sq = self.prev_centers[original_prev_idx, 3].float()
                    
                    if min_dist < prev_radius_sq:
                        matched_id = self.prev_centers[original_prev_idx, 2].to(new_panoptic_map.dtype)
                        new_panoptic_map[center_mask] = matched_id
                        center[2] = matched_id
                        prev_matched[original_prev_idx] = True
                        match_found = True
                        
            if not match_found:
                new_id = (sem_id * self.label_divisor + self.next_tracking_id).to(new_panoptic_map.dtype)
                new_panoptic_map[center_mask] = new_id
                center[2] = new_id
                self.next_tracking_id += 1
                
            updated_current_centers.append(center)
            
        updated_current_centers = torch.stack(updated_current_centers)
        
        # Keep unmatched previous centers for 'sigma' frames in case they reappear
        unmatched_prev = self.prev_centers[~prev_matched].clone()
        final_centers = torch.cat([updated_current_centers, unmatched_prev], dim=0) if len(unmatched_prev) > 0 else updated_current_centers
            
        if len(final_centers) > 0:
            final_centers[:, 4] += 1 # Age up all centers
            final_centers = final_centers[final_centers[:, 4] <= self.sigma_track] # Kill old ones
            
        self.prev_centers = final_centers
        return new_panoptic_map