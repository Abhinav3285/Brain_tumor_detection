import streamlit as st
import sys
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# -------------------------------------------------
# 1. SETUP PATHS & IMPORT SEGMENTATION
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

SEGMENTATION_AVAILABLE = False
GRADCAM_AVAILABLE = False
UNET_PATH = os.path.join(PROJECT_ROOT, "model", "segmentation", "unet_best.pth")

if os.path.exists(UNET_PATH):
    try:
        from model.segmentation.predict_unet import predict_mask
        from model.segmentation.grad_cam_unet import grad_cam_unet, generate_gradcam_overlay
        from model.segmentation.unet_model import UNet
        import torch
        SEGMENTATION_AVAILABLE = True
        GRADCAM_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not import segmentation modules: {e}")
        SEGMENTATION_AVAILABLE = False
        GRADCAM_AVAILABLE = False

# -------------------------------------------------
# 2. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="NeuroScan AI", layout="wide", page_icon="🧠")

# -------------------------------------------------
# 3. LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_yolo():
    path = os.path.join(PROJECT_ROOT, "results", "brain_tumor_run", "weights", "best.pt")
    return YOLO(path) if os.path.exists(path) else None

@st.cache_resource
def load_unet():
    if not SEGMENTATION_AVAILABLE:
        return None
    try:
        device = torch.device("cpu")
        model = UNet().to(device)
        model.load_state_dict(torch.load(UNET_PATH, map_location=device, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading U-Net: {e}")
        return None

model = load_yolo()
unet_model = load_unet() if SEGMENTATION_AVAILABLE else None

# -------------------------------------------------
# 4. UTILITY FUNCTIONS
# -------------------------------------------------
def generate_yolo_cam(image_np, results):
    """Generate YOLO-based heatmap from detection boxes"""
    heatmap = np.zeros(image_np.shape[:2], dtype=np.float32)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            heatmap[y1:y2, x1:x2] += conf
    if np.max(heatmap) > 0:
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return image_np

def generate_unet_gradcam(image_np, unet_model):
    """Generate Grad-CAM from U-Net model"""
    try:
        img = cv2.resize(image_np, (256, 256))
        img = img / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        
        cam = grad_cam_unet(unet_model, img_tensor)
        overlay = generate_gradcam_overlay(image_np, cam)
        return overlay
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return image_np

def determine_tumor_stage(area_pixels, tumor_type, image_dimensions):
    """
    Enhanced tumor staging based on multiple criteria.
    
    Args:
        area_pixels: Tumor area in pixels
        tumor_type: Type of tumor (glioma, meningioma, pituitary)
        image_dimensions: Tuple of (width, height) of the image
    
    Returns:
        Tuple of (stage_text, severity_color, description)
    """
    # Calculate percentage of image occupied by tumor
    total_image_area = image_dimensions[0] * image_dimensions[1]
    tumor_percentage = (area_pixels / total_image_area) * 100
    
    # Staging logic based on area and percentage
    if area_pixels < 3000 or tumor_percentage < 2:
        stage = "Stage I (Early/Low Grade)"
        color = "#28a745"  # Green
        description = "Small tumor detected. Early intervention recommended."
    elif area_pixels < 8000 or tumor_percentage < 5:
        stage = "Stage II (Moderate/Intermediate)"
        color = "#ffc107"  # Yellow
        description = "Moderate-sized tumor. Further evaluation needed."
    elif area_pixels < 15000 or tumor_percentage < 10:
        stage = "Stage III (Advanced/High Grade)"
        color = "#fd7e14"  # Orange
        description = "Large tumor detected. Immediate medical attention advised."
    else:
        stage = "Stage IV (Severe/Critical)"
        color = "#dc3545"  # Red
        description = "Very large tumor. Urgent intervention required."
    
    return stage, color, description

# -------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    conf_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)
    st.divider()
    st.info(f"🧬 Segmentation Engine: {'✅ ACTIVE' if SEGMENTATION_AVAILABLE else '❌ INACTIVE'}")
    st.info(f"🔥 Grad-CAM Engine: {'✅ ACTIVE' if GRADCAM_AVAILABLE else '❌ INACTIVE'}")
    if model:
        st.write("**Detectable Tumor Types:**")
        for i, name in model.names.items():
            if name != "no_tumor":
                st.code(f"{i}: {name.upper()}")

# -------------------------------------------------
# 6. MAIN UI
# -------------------------------------------------
st.title("🧠 NeuroScan Vision: Brain Tumor Analysis")
st.markdown("**AI-Powered Brain Tumor Detection, Segmentation, and Staging System**")
st.markdown("---")

uploaded_files = st.file_uploader("Upload MRI Scans", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and model:
    # Use tabs for multiple files
    tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files])
    
    for i, uploaded_file in enumerate(uploaded_files):
        with tabs[i]:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)
            img_dimensions = (img_np.shape[1], img_np.shape[0])
            
            # YOLO Prediction
            results = model.predict(img_np, conf=conf_threshold)
            res = results[0]
            
            col_viz, col_data = st.columns([2, 1])
            
            with col_viz:
                st.subheader("📊 Analysis Visualizations")
                viz_tabs = st.tabs(["🎯 Detection", "🔥 Grad-CAM", "🎨 Segmentation"])
                
                with viz_tabs[0]:
                    st.image(res.plot(), use_container_width=True, caption="YOLOv8 Object Detection")
                
                with viz_tabs[1]:
                    if GRADCAM_AVAILABLE and unet_model is not None:
                        # Use U-Net Grad-CAM
                        gradcam_img = generate_unet_gradcam(img_np, unet_model)
                        st.image(gradcam_img, use_container_width=True, caption="U-Net Grad-CAM Focus Area")
                    else:
                        # Fallback to YOLO-based heatmap
                        st.image(generate_yolo_cam(img_np, results), use_container_width=True, caption="YOLO-based Focus Area")
                        if not GRADCAM_AVAILABLE:
                            st.info("💡 Using YOLO-based visualization (U-Net Grad-CAM not available)")
                
                with viz_tabs[2]:
                    if SEGMENTATION_AVAILABLE:
                        # REAL U-NET SEGMENTATION
                        try:
                            mask = predict_mask(img_np)
                            overlay = img_np.copy()
                            overlay[mask > 0] = [255, 0, 0]
                            st.image(cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0), 
                                   use_container_width=True, caption="U-Net Tumor Segmentation")
                        except Exception as e:
                            st.error(f"Segmentation error: {e}")
                    elif res.boxes:
                        # SIMULATED SEGMENTATION (Using YOLO boxes)
                        st.info("💡 Using AI-Simulated Boundary (YOLO-based)")
                        overlay = img_np.copy()
                        for box in res.boxes:
                            # Get the box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Create a rounded shape inside the box to look like a tumor mask
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                            cv2.ellipse(overlay, center, axes, 0, 0, 360, (255, 0, 0), -1)
                        
                        # Add a "fuzzy" blur to make it look like a real segmentation mask
                        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
                        st.image(cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0), use_container_width=True)
                    else:
                        st.warning("No tumor detected to segment.")
            
            with col_data:
                st.subheader("🧪 Tumor Analysis Report")
                if res.boxes and len(res.boxes) > 0:
                    tumor_detected = False
                    for j, box in enumerate(res.boxes):
                        cid = int(box.cls[0])
                        label = model.names[cid]
                        
                        # Skip if it's "no_tumor" class (case-insensitive)
                        if label.lower() == "no_tumor":
                            continue
                        
                        tumor_detected = True
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0]
                        area = int((x2 - x1) * (y2 - y1))
                        
                        # ENHANCED STAGING LOGIC
                        stage, color, description = determine_tumor_stage(area, label, img_dimensions)

                        # Detail Card
                        with st.container(border=True):
                            st.markdown(f"### 🧬 Tumor {j+1}: {label.upper()}")
                            st.markdown(f"**Estimated Stage:** <span style='color:{color}; font-weight:bold'>{stage}</span>", 
                                      unsafe_allow_html=True)
                            st.caption(description)
                            st.write(f"**Area:** {area:,} pixels²")
                            st.write(f"**Confidence:** {conf:.1%}")
                            st.progress(conf)
                            
                            # Calculate tumor percentage
                            tumor_pct = (area / (img_dimensions[0] * img_dimensions[1])) * 100
                            st.write(f"**Image Coverage:** {tumor_pct:.2f}%")
                    
                    if not tumor_detected:
                        st.success("✅ No tumor detected in this scan.")
                else:
                    st.success("✅ No abnormalities detected.")

st.divider()
st.caption("⚠️ **Clinical Disclaimer:** This is an AI-assisted research tool and should not be used for final medical diagnosis. "
          "Always consult with qualified healthcare professionals for medical decisions.")


