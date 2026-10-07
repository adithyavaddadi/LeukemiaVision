import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    """
    Standard PyTorch Grad-CAM for CNN backbones and Hybrid models.
    Hooks into activations and gradients of the target feature layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=1):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(input_tensor)
        score = logits[:, target_class]
        score.backward(retain_graph=True)
        
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - torch.min(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        cam_np = cam.squeeze().cpu().numpy()
        return cam_np

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def numpy_gap_gradcam(feature_maps_np, classifier_weights_np, target_class=1):
    """
    Closed-Form NumPy Grad-CAM for GAP Models (EfficientNet-B0, ResNet-50).
    Requires zero PyTorch dependency; runs entirely inside ONNX Runtime CPU.
    feature_maps_np: shape (1, C, H, W)
    classifier_weights_np: shape (num_classes, C)
    """
    weights = classifier_weights_np[target_class]  # shape (C,)
    # Reshape weights to (C, 1, 1)
    weights = weights[:, np.newaxis, np.newaxis]
    f = feature_maps_np[0]  # (C, H, W)
    
    cam = np.sum(weights * f, axis=0)  # (H, W)
    cam = np.maximum(cam, 0)  # ReLU
    
    cam_min = np.min(cam)
    cam_max = np.max(cam)
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)

def generate_overlay(original_pil, heatmap_np, alpha=0.4, colormap="jet"):
    """
    Overlays a float32 heatmap [0, 1] onto the original PIL image.
    """
    w, h = original_pil.size
    heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize((w, h), Image.Resampling.BILINEAR)
    heatmap_norm = np.array(heatmap_resized) / 255.0
    
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = (cmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    
    orig_np = np.array(original_pil)
    overlay = (orig_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap_rgb)

def sanity_check_randomized_weights(model_class, input_tensor, target_class=1):
    """
    Adebayo et al. Model Parameter Randomization Test.
    Ensures that Grad-CAM reflects trained parameters rather than edge artifacts.
    """
    untrained_model = model_class(num_classes=2, pretrained=False)
    layer = untrained_model.get_last_conv_layer()
    cam_engine = GradCAM(untrained_model, layer)
    cam_untrained = cam_engine.generate_heatmap(input_tensor, target_class=target_class)
    cam_engine.remove_hooks()
    return cam_untrained
