import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from   torchvision.models import resnet50, ResNet50_Weights
from   torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from   post_processor import PanopticPostProcessor, MotionTracker

class MotionDeepLabResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet50(weights=None)
        print("Loading COCO segmentation weights from Torchvision...")

        deeplab_coco = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        backbone_weights = deeplab_coco.backbone.state_dict()
        resnet.load_state_dict(backbone_weights, strict=False)

        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=7, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        with torch.no_grad():
            new_weights = torch.zeros(
                original_conv1.out_channels, 7, 
                original_conv1.kernel_size[0], 
                original_conv1.kernel_size[1]
            )
            # Use the newly loaded COCO weights for the RGB channels
            new_weights[:, :3, :, :] = original_conv1.weight / 2.0
            new_weights[:, 3:6, :, :] = original_conv1.weight / 2.0
            self.conv1.weight.copy_(new_weights)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(self.bn1(x)))
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)

        return {'res2': res2, 'res3': res3, 'res5': res5}

class ConvBNReLU(nn.Module):
    """A standard block grouping Conv2d, BatchNorm2d, and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = ConvBNReLU(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        h, w = x.shape[2:]
        x = self.pool(x)
        x = self.proj(x)

        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[3, 6, 9]):
        super().__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels, kernel_size=1)

        self.branch2 = ConvBNReLU(in_channels, out_channels, kernel_size=3, 
                                  padding=atrous_rates[0], dilation=atrous_rates[0])
        self.branch3 = ConvBNReLU(in_channels, out_channels, kernel_size=3, 
                                  padding=atrous_rates[1], dilation=atrous_rates[1])
        self.branch4 = ConvBNReLU(in_channels, out_channels, kernel_size=3, 
                                  padding=atrous_rates[2], dilation=atrous_rates[2])
    
        self.branch5 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(ConvBNReLU(out_channels * 5, out_channels, kernel_size=1),
                                     nn.Dropout(0.1))
    def forward(self, x):
        res = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        return self.project(res)

# Test
aspp = ASPP(in_channels=2048, out_channels=256)
dummy_res5 = torch.randn(2, 2048, 12, 39)
output = aspp(dummy_res5)
print("ASPP Test")
print(f"Input shape:  {dummy_res5.shape}")
print(f"Output shape: {output.shape}")

class PanopticDeepLabSingleDecoder(nn.Module):
    """Decodes features for a single task (Semantic or Instance)."""

    def __init__(self, high_level_feature_name='res5',
                 low_level_feature_names=['res3', 'res2'],
                 low_level_channels_project=[64, 32],
                 aspp_channels=256,
                 decoder_channels=256,
                 atrous_rates=[3, 6, 9]):
        # This calls the __init__ of the upper class nn.Module
        super().__init__()

        self.high_level_feature_name = high_level_feature_name
        self.low_level_feature_names = low_level_feature_names

        self.aspp = ASPP(in_channels=2048, out_channels=aspp_channels, atrous_rates=atrous_rates)

        self.project_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        resnet_out_channels = {'res3': 512, 'res2': 256}
        previous_channels = aspp_channels

        for i, feature_name in enumerate(low_level_feature_names):
            proj_channels = low_level_channels_project[i]

            self.project_convs.append(
                ConvBNReLU(resnet_out_channels[feature_name], proj_channels, kernel_size=1)
            )

            fusion_in_channels = previous_channels + proj_channels
            self.fusion_convs.append(
                ConvBNReLU(fusion_in_channels, decoder_channels, kernel_size=5, padding=2)
            )
            previous_channels = decoder_channels
    
    def forward(self, features):
        high_level_feat = features[self.high_level_feature_name]
        combined_features = self.aspp(high_level_feat)

        for i, feature_name in enumerate(self.low_level_feature_names):
            low_level_feat = features[feature_name]
            low_level_feat = self.project_convs[i](low_level_feat)

            h, w = low_level_feat.shape[2:]
            combined_features = F.interpolate(
                combined_features, size=(h, w), mode='bilinear', align_corners=False
            )
            combined_features = torch.cat([combined_features, low_level_feat], dim=1)

            combined_features = self.fusion_convs[i](combined_features)
        
        return combined_features

class PanopticDeepLabSingleHead(nn.Module):
    """A single prediction head for Panoptic/Motion-DeepLab."""
    
    def __init__(self, in_channels, intermediate_channels, output_channels):
        super().__init__()

        self.conv_block = ConvBNReLU(in_channels, intermediate_channels, kernel_size=5, padding=2)
        self.final_conv = nn.Conv2d(intermediate_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv_block(x)
        return self.final_conv(x)
    
class MotionDeepLabDecoder(nn.Module):
    """The full Motion-DeepLab dual-decoder and multi-head architecture."""

    def __init__(self):
        super().__init__()

        # All values are taken from the proto files of the DeepLab library
        # configs/kitti/motion_deeplab/resnet50_os32_trainval.textproto
        # and model.proto
        decoder_channels = 256
        self.semantic_decoder = PanopticDeepLabSingleDecoder(
            high_level_feature_name='res5',
            low_level_feature_names=['res3', 'res2'],
            low_level_channels_project=[64, 32],
            aspp_channels=256,
            decoder_channels=decoder_channels,
            atrous_rates=[3, 6, 9]
        )
        # Predicst the 19 semantic classes
        self.semantic_head = PanopticDeepLabSingleHead(
            in_channels=decoder_channels,
            intermediate_channels=256,
            output_channels=19
        )

        instance_decoder_channels = 128
        self.instance_decoder = PanopticDeepLabSingleDecoder(
            high_level_feature_name='res5',
            low_level_feature_names=['res3', 'res2'],
            low_level_channels_project=[32, 16],
            aspp_channels=256,
            decoder_channels=instance_decoder_channels,
            atrous_rates=[3, 6, 9]
        )

        # Predicts the center of each object (1 channel heatmap)
        self.instance_center_head = PanopticDeepLabSingleHead(
            in_channels=instance_decoder_channels, 
            intermediate_channels=32, 
            output_channels=1
        )

        # Predicts the (Y, X) offset from each pixel to its object center (2 channels)
        self.intance_regression_head = PanopticDeepLabSingleHead(
            in_channels=instance_decoder_channels, 
            intermediate_channels=32, 
            output_channels=2
        )

        # Predicts the (Y, X) offset from each pixel to its center in the previous frame
        self.motion_regression_head = PanopticDeepLabSingleHead(
            in_channels=instance_decoder_channels, 
            intermediate_channels=32, 
            output_channels=2
        )
    def forward(self, features):
        results = {}

        semantic_features = self.semantic_decoder(features)
        results['semantic_logits'] = self.semantic_head(semantic_features)

        instance_features = self.instance_decoder(features)
        results['center_heatmap'] = self.instance_center_head(instance_features)
        results['center_offsets'] = self.intance_regression_head(instance_features)
        results['motion_offsets'] = self.motion_regression_head(instance_features)

        return results
    
class MotionDeepLab(nn.Module):
    def __init__(self, thing_class_ids, label_divisor=1000, void_label=255):
        super().__init__()

        self.encoder = MotionDeepLabResNet50()
        self.decoder = MotionDeepLabDecoder()

        self.post_processor = PanopticPostProcessor(
            thing_class_ids=thing_class_ids, 
            label_divisor=label_divisor,
            void_label=void_label
        )
        self.tracker = MotionTracker(
            label_divisor=label_divisor, 
            void_label=void_label
        )
        self.prev_center_heatmap = None

    def forward(self, x, gt_prev_heatmap=None):
        B, C, H, W = x.shape
        frame1, frame2 = torch.split(x, [3, 3], dim=1)

        if self.training:
            if gt_prev_heatmap is None:
                raise ValueError("Training requires gt_prev_heatmap for the 7th channel.")
            x_7ch = torch.cat([x, gt_prev_heatmap], dim=1)
        else:
            frame1, frame2 = torch.split(x, [3, 3], dim=1)
            if torch.all(frame1 == frame2) or self.prev_center_heatmap is None:
                self.prev_center_heatmap = torch.zeros((B, 1, H, W), dtype=x.dtype, device=x.device)
            x_7ch = torch.cat([x, self.prev_center_heatmap], dim=1)

        features = self.encoder(x_7ch)
        results = self.decoder(features)

        for key in results.keys():
            results[key] = F.interpolate(
                results[key], size=(H, W), mode='bilinear', align_corners=False
            )

            if 'offset' in key:
                scale_y = H / (H // 4)
                scale_x = W / (W // 4)
                results[key][:, 0, :, :] *= scale_y
                results[key][:, 1, :, :] *= scale_x
        
        if self.training:
            return results
        
        panoptic_map, instance_map = self.post_processor(
            semantic_logits=results['semantic_logits'],
            center_heatmap=results['center_heatmap'],
            offset_map=results['center_offsets']
        )
        
        results['instance_pred'] = instance_map
        
        if B == 1:
            next_heatmap, current_centers = self.tracker.render_panoptic_map_as_heatmap(panoptic_map[0])
            
            tracked_panoptic_map = self.tracker.assign_instances_to_previous_tracks(
                current_centers=current_centers,
                heatmap=next_heatmap,
                offsets=results['motion_offsets'][0],
                panoptic_map=panoptic_map[0]
            )
            results['panoptic_pred'] = tracked_panoptic_map.unsqueeze(0) 
            self.prev_center_heatmap = next_heatmap.detach()
        else:
            results['panoptic_pred'] = panoptic_map # Fallback if B > 1

        if 'center_heatmap' in results:
            results['center_heatmap'] = results['center_heatmap'].squeeze(1)

        return results
    
    def update_tracking_state(self, new_center_heatmap):
        self.prev_center = new_center_heatmap.detach().unsqueeze(1)

    def reset_tracking_state(self):
        self.prev_center_heatmap = None
        device = next(self.parameters()).device
        self.tracker.reset_state(device)
