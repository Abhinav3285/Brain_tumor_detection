# model/segmentation/predict_unet.py
import torch
import numpy as np
import cv2
import os
from .unet_model import UNet   # ✅ RELATIVE IMPORT

# Get the absolute path to the model file
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "unet_best.pth")

device = torch.device("cpu")

model = UNet().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
else:
    print(f"Warning: U-Net model not found at {MODEL_PATH}")

def predict_mask(image_np):
    img = cv2.resize(image_np, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        mask = model(img)[0][0].numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

    return mask
