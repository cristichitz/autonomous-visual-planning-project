import  torch
import  numpy as np
import  cv2
import  matplotlib.pyplot as plt
from    torch.amp import autocast
from    dataset import KittiStepDataset
from    torch.utils.data import DataLoader
from    model import MotionDeepLab
from    stq   import STQuality
import  os
import  sys
from matplotlib.colors import ListedColormap

# # Official Cityscapes / KITTI-STEP RGB colors (normalized to 0.0 - 1.0)
cityscapes_colors = [
    [128/255,  64/255, 128/255],  # 0: road (Dark Purple)
    [244/255,  35/255, 232/255],  # 1: sidewalk (Magenta/Pink)
    [ 70/255,  70/255,  70/255],  # 2: building (Dark Grey)
    [102/255, 102/255, 156/255],  # 3: wall (Slate/Grey-Blue)
    [190/255, 153/255, 153/255],  # 4: fence (Dusty Rose/Light Brown)
    [153/255, 153/255, 153/255],  # 5: pole (Grey)
    [250/255, 170/255,  30/255],  # 6: traffic light (Orange)
    [220/255, 220/255,   0/255],  # 7: traffic sign (Yellow)
    [107/255, 142/255,  35/255],  # 8: vegetation (Olive Green)
    [152/255, 251/255, 152/255],  # 9: terrain (Light Green)
    [ 70/255, 130/255, 180/255],  # 10: sky (Steel Blue)
    [220/255,  20/255,  60/255],  # 11: person (Crimson/Red)
    [255/255,   0/255,   0/255],  # 12: rider (Bright Red)
    [  0/255,   0/255, 142/255],  # 13: car (Dark Blue)
    [  0/255,   0/255,  70/255],  # 14: truck (Navy Blue)
    [  0/255,  60/255, 100/255],  # 15: bus (Dark Teal)
    [  0/255,  80/255, 100/255],  # 16: train (Turquoise Blue)
    [  0/255,   0/255, 230/255],  # 17: motorcycle (Blue)
    [119/255,  11/255,  32/255],  # 18: bicycle (Maroon/Dark Red)
    [  0/255,   0/255,   0/255],  # 19: void (Black)
]

cityscapes_cmap = ListedColormap(cityscapes_colors)
cityscapes_colors_255 = (np.array(cityscapes_colors) * 255).astype(np.uint8)

def colorize_panoptic(panoptic_map, label_divisor=1000):
    """
    Colors background classes using standard Cityscapes palettes, 
    and tracked instances using persistent random colors.
    """
    h, w = panoptic_map.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_ids = np.unique(panoptic_map)
    for pan_id in unique_ids:
        sem_id = pan_id // label_divisor
        inst_id = pan_id % label_divisor
        
        mask = (panoptic_map == pan_id)
        
        if sem_id == 255 or sem_id >= 19: 
            color = [0, 0, 0]
        elif inst_id == 0:
            color = cityscapes_colors_255[sem_id]
        else:
            rng = np.random.RandomState(pan_id)
            color = rng.randint(50, 255, size=3) 
            
        rgb_map[mask] = color
        
    return rgb_map

def visualize_prediction(image, predictions):
    """Renders a side-by-side comparison of the raw input and the panoptic tracking."""
    img_rgb = image.permute(1, 2, 0).cpu().numpy()
    img_rgb = np.clip(img_rgb, 0, 1)
    img_rgb_255 = (img_rgb * 255).astype(np.uint8)

    panoptic_tensor = predictions['panoptic_pred'][0].cpu().numpy()
    panoptic_rgb = colorize_panoptic(panoptic_tensor, label_divisor=1000)

    # Blends 40% of the original image with 60% of the prediction map
    blended = cv2.addWeighted(img_rgb_255, 0.4, panoptic_rgb, 0.6, 0)

    unique_ids = np.unique(panoptic_tensor)
    for pan_id in unique_ids:
        sem_id = pan_id // 1000
        inst_id = pan_id % 1000
        if inst_id > 0 and sem_id < 19 and sem_id != 255:
            y_coords, x_coords = np.where(panoptic_tensor == pan_id)

            if len(x_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                text = str(inst_id)

                cv2.putText(blended, text, (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),3)
                cv2.putText(blended, text, (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Frame", fontsize=18)
    axes[0].axis('off')
    
    axes[1].imshow(blended)
    axes[1].set_title("Persistent Panoptic Tracking", fontsize=18)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def fig_to_frame(fig):
    """Converts a Matplotlib figure to a BGR numpy array for OpenCV."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_rgba = np.asarray(buf)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    return img_bgr

class MotionDeepLabEvaluator:
    def __init__(self, weights_path, kitti_root='.'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kitti_root = kitti_root
        self.thing_ids = [11, 13]

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

        self._load_model(weights_path)
    
    def _load_model(self, path):
        self.model = MotionDeepLab(thing_class_ids=self.thing_ids, label_divisor=1000).to(self.device)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Model loaded from {path}")
        else:
            print(f"Error: Model weights not found at '{path}'.")
            sys.exit(1)
        self.model.eval()
    
    def _get_start_idx(self, dataset, target_seq):
        for idx, sample in enumerate(dataset.samples):
            if sample['sequence_id'] == target_seq:
                return idx
        return None
    
    def evaluate_sequence(self, target_seq="0001", split='val', num_frames=350, fps=10, out_name='outputs/evaluation_video.mp4'):
        dataset = KittiStepDataset(root_dir=self.kitti_root, split=split)
        start_idx = self._get_start_idx(dataset, target_seq)

        if start_idx is None:
            print(f"Error: Sequence {target_seq} not found in {split} set!")
            return
        
        has_gt = (split == 'val')
        if has_gt:
            stq_metric = STQuality(
                num_classes=19, things_list=self.thing_ids, 
                ignore_label=255, max_instances_per_category=1000, offset=2**32
            )
        
        video_writer = None
        print(f"Starting inference on {target_seq} ({split} split)...")
        
        with torch.no_grad():
            for i in range(start_idx, min(start_idx + num_frames, len(dataset))):
                if has_gt:
                    stacked_images, y_true_sem, y_true_inst, _ = dataset[i]
                else:
                    stacked_images = dataset[i]

                images_6ch = stacked_images.unsqueeze(0).to(self.device)
                with autocast(device_type='cuda'):
                    predictions = self.model(images_6ch)

                if has_gt:
                    y_pred = predictions['panoptic_pred'].squeeze(0)
                    y_true = ((y_true_sem * 1000) + y_true_inst).to(self.device)
                    stq_metric.update_state(y_true=y_true, y_pred=y_pred, sequence_id=target_seq)
                
                curr_rgb = torch.clamp((images_6ch[0, :3, :, :] * self.std) + self.mean, 0, 1)
                fig = visualize_prediction(curr_rgb, predictions)
                frame_bgr = fig_to_frame(fig)
                
                if video_writer is None:
                    h, w, _ = frame_bgr.shape
                    os.makedirs(os.path.dirname(out_name) or '.', exist_ok=True)
                    video_writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                video_writer.write(frame_bgr)
                plt.close(fig) 
                
                print(f"Processed frame {i - start_idx + 1}/{num_frames}")

        if video_writer:
            video_writer.release()
        print(f"✓ Video saved as {out_name}")

        if has_gt:
            final_scores = stq_metric.result()
            print("\nFinal Evaluation Scores:")
            print(f"STQ: {final_scores['STQ']:.4f}")
            print(f"AQ:  {final_scores['AQ']:.4f}")
            print(f"SQ:  {final_scores['IoU']:.4f}")
        else:
            print(f"\nEvaluation: Ran on '{split}' split (No Ground Truth). Skipping STQ scores.")

if __name__ == '__main__':
    evaluator = MotionDeepLabEvaluator(
        weights_path='weights/motion_deeplab_epoch_90.pth', 
        kitti_root='.'
    )
    
    # Run Validation sequence
    evaluator.evaluate_sequence(
        target_seq="0029", 
        split="test", 
        num_frames=200, 
        out_name="outputs/teak_epoch90.mp4"
    )
    
    # Example: Run Test sequence
    # evaluator.evaluate_sequence(
    #     target_seq="0005", 
    #     split="test", 
    #     num_frames=100, 
    #     out_name="outputs/test_video.mp4"
    # )