import  torch
import  numpy as np
import  cv2
import  matplotlib.pyplot as plt
from    torch.amp import autocast
from    dataset import KittiStepDataset
from    torch.utils.data import DataLoader
from    model import MotionDeepLab
import  os
import  sys
from matplotlib.colors import ListedColormap

# Official Cityscapes / KITTI-STEP RGB colors (normalized to 0.0 - 1.0)
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

def visualize_prediction(image, predictions):
    # 1. Prepare Semantic Map
    sem_logits = predictions['semantic_logits'][0].cpu().numpy()
    sem_pred = np.argmax(sem_logits, axis=0)
    sem_pred[sem_pred == 255] = 19

    
    # 2. Prepare Center Heatmap
    center_heat = torch.sigmoid(predictions['center_heatmap'][0, 0]).cpu().numpy()
    
    # 3. Prepare Motion Offsets (as HSV Optical Flow)
    motion_yx = predictions['motion_offsets'][0].cpu().numpy()
    mag, ang = cv2.cartToPolar(motion_yx[1], motion_yx[0])
    hsv = np.zeros((motion_yx.shape[1], motion_yx.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    motion_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    axes[0, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Input Frame")
    
    axes[0, 1].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].imshow(sem_pred, cmap=cityscapes_cmap, alpha=0.5, vmin=0, vmax=19)
    axes[0, 1].set_title("Semantic Overlay")
    
    axes[1, 0].imshow(center_heat, cmap='magma')
    axes[1, 0].set_title("Instance Center Heatmap")
    
    axes[1, 1].imshow(motion_rgb)
    axes[1, 1].set_title("Motion Vectors (HSV)")
    
    plt.tight_layout()
    return fig

def fig_to_frame(fig):
    """Converts a Matplotlib figure to a BGR numpy array for OpenCV."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_rgba = np.asarray(buf)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    return img_bgr


BATCH_SIZE = 4
KITTI_STEP_ROOT = '.'
MODEL_SAVE_PATH = 'weights/motion_deeplab_epoch_11.pth'

val_ds = KittiStepDataset(root_dir=KITTI_STEP_ROOT, split='val')
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MotionDeepLab().to(device)

if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(f"✓ Model loaded from {MODEL_SAVE_PATH}")
else:
    print(f"Error: Model weights not found at '{MODEL_SAVE_PATH}'.")
    sys.exit(1)

print("Starting Video Evaluation...")
model.eval()

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

TARGET_SEQ = "0013"
NUM_FRAMES = 250
FPS = 5
video_writer = None

start_idx = None
for idx, sample in enumerate(val_ds.samples):
    if sample['sequence_id'] == TARGET_SEQ:
        start_idx = idx
        break

# Bootstrapping the network's memory: Frame 1 has no previous heatmap, so it's blank.
prev_predicted_heatmap = torch.zeros((1, 1, 384, 1248), device=device)
if start_idx is None:
    print(f"Error: Sequence {TARGET_SEQ} not found in validation set!")
else:
    print(f"Found {TARGET_SEQ} at index {start_idx}. Starting video generation...")
    with torch.no_grad():
        for i in range(start_idx, start_idx + NUM_FRAMES):
            # We access val_ds directly to guarantee we get frames in chronological order
            stacked_images, _, _, _ = val_ds[i]
            
            # Add the batch dimension (Batch=1)
            images = stacked_images.unsqueeze(0).to(device)
            
            # Concatenate 6 RGB channels with the 1 heatmap channel from the PREVIOUS loop
            model_input = torch.cat([images, prev_predicted_heatmap], dim=1)
            
            # Forward pass
            with autocast('cuda'):
                predictions = model(model_input)
                
            # --- The Crucial Tracking Step ---
            # Save this frame's center prediction so the NEXT frame can look at it
            # prev_predicted_heatmap = torch.sigmoid(predictions['center_heatmap']).detach()
            
            # Un-normalize the RGB image for Matplotlib
            curr_rgb = images[0, :3, :, :]
            curr_rgb = curr_rgb * std + mean
            curr_rgb = torch.clamp(curr_rgb, 0, 1)
            
            # Draw the 2x2 grid
            fig = visualize_prediction(curr_rgb, predictions)
            frame_bgr = fig_to_frame(fig)
            
            # Initialize the OpenCV VideoWriter on the first frame once we know the exact pixel dimensions
            if video_writer is None:
                h, w, _ = frame_bgr.shape
                video_writer = cv2.VideoWriter('outputs/evaluation_video2.mp4', 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            FPS, (w, h))
                
            video_writer.write(frame_bgr)
            
            # IMPORTANT: Close the figure to prevent your RAM from exploding
            plt.close(fig) 
            
            print(f"Processed frame {i + 1}/{start_idx + NUM_FRAMES}")

# Save the file
if video_writer:
    video_writer.release()
print("✓ Video saved as evaluation_video.mp4")