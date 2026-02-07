import torch
import cv2
import numpy as np
import os

def grad_cam_unet(model, image_tensor, target_layer='enc1'):
    """
    Generate Grad-CAM visualization for U-Net model.
    
    Args:
        model: The U-Net model
        image_tensor: Input image tensor
        target_layer: Name of the layer to visualize (default: 'enc1')
    
    Returns:
        cam: Grad-CAM heatmap as numpy array
    """
    feature_maps = []
    gradients = []

    def save_features(module, input, output):
        feature_maps.append(output)

    def save_grads(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on the target layer
    target = getattr(model, target_layer, None)
    if target is None:
        print(f"Warning: Layer {target_layer} not found, using enc1")
        target = model.enc1
    
    hook_forward = target.register_forward_hook(save_features)
    hook_backward = target.register_full_backward_hook(save_grads)

    # Forward pass
    model.eval()
    image_tensor.requires_grad_(True)
    output = model(image_tensor)
    
    # Backward pass - use the mean of the output as the target
    output.mean().backward()

    # Get the feature maps and gradients
    if len(feature_maps) > 0 and len(gradients) > 0:
        fmap = feature_maps[0][0]  # First batch item
        grad = gradients[0][0]  # First batch item

        # Calculate weights
        weights = grad.mean(dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]

        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (256, 256))
        cam = (cam * 255).astype(np.uint8)
    else:
        # Return empty heatmap if hooks didn't capture anything
        cam = np.zeros((256, 256), dtype=np.uint8)

    # Remove hooks
    hook_forward.remove()
    hook_backward.remove()

    return cam


def generate_gradcam_overlay(image_np, cam):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image_np: Original image as numpy array (H, W, 3)
        cam: Grad-CAM heatmap (H, W) with values 0-255
    
    Returns:
        overlay: Image with Grad-CAM overlay
    """
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # Overlay on original image
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    
    return overlay
