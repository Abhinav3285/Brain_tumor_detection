# 🧠 Brain Tumor Detection System - Complete Setup Guide

## Overview
This AI-powered system provides comprehensive brain tumor analysis including:
1. **Tumor Detection** - Identifies and locates tumors in MRI scans
2. **Tumor Staging** - Estimates the stage/severity of detected tumors
3. **Segmentation** - Shows precise tumor boundaries
4. **Grad-CAM Visualization** - Highlights areas the AI is focusing on

---

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Software Dependencies
All dependencies are listed in `requirements.txt`:
- ultralytics (YOLOv8 framework)
- torch & torchvision (Deep learning backend)
- streamlit (Web interface)
- opencv-python-headless (Image processing)
- pillow, numpy (Image manipulation)
- reportlab (PDF generation)

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abhinav3285/Brain_tumor_detection.git
cd Brain_tumor_detection
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- YOLOv8 for object detection
- PyTorch for deep learning
- Streamlit for the web interface
- OpenCV for image processing
- Other required libraries

### Step 4: Verify Model Files
Check that you have the trained models:

```bash
# YOLO model should be at:
results/brain_tumor_run/weights/best.pt

# U-Net segmentation model should be at:
model/segmentation/unet_best.pth
```

If models are missing, you'll need to train them (see Training section below).

### Step 5: Run the Application
```bash
# Navigate to the app directory
cd app

# Launch the Streamlit web interface
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

---

## 🎯 How to Use the System

### Using the Web Interface

1. **Launch the App**
   ```bash
   cd app
   streamlit run app.py
   ```

2. **Upload MRI Images**
   - Click "Browse files" or drag & drop MRI scan images
   - Supported formats: JPG, PNG, JPEG
   - Can upload multiple images at once

3. **Adjust Settings (Sidebar)**
   - **Detection Confidence**: Controls how certain the AI must be (0.4 = 40% confident)
   - Lower values detect more but may have false positives
   - Higher values are more conservative

4. **View Results**
   The system provides three visualization tabs:

   **🎯 Detection Tab**
   - Shows bounding boxes around detected tumors
   - Color-coded by tumor type (Glioma, Meningioma, Pituitary)
   - Displays confidence percentage

   **🔥 Grad-CAM Tab**
   - Heatmap showing where the AI is "looking"
   - Red/yellow areas = high focus regions
   - Blue/green areas = less important regions
   - Uses U-Net model if available, YOLO fallback otherwise

   **🎨 Segmentation Tab**
   - Precise tumor boundary overlay in red
   - Uses U-Net segmentation model if available
   - Falls back to ellipse-based approximation from YOLO boxes

5. **Understand the Analysis Report**
   - **Tumor Type**: Glioma, Meningioma, or Pituitary
   - **Estimated Stage**: 
     - Stage I (Early/Low Grade) - Small, treatable
     - Stage II (Moderate) - Growing, needs attention
     - Stage III (Advanced) - Large, serious
     - Stage IV (Severe/Critical) - Very large, urgent
   - **Area**: Size in pixels squared
   - **Confidence**: How certain the AI is
   - **Image Coverage**: What % of the scan is tumor

---

## 🔧 Training Models (Optional)

If you need to train or retrain the models:

### Training YOLOv8 Detection Model

```bash
# From the project root directory
python train.py
```

Or train directly:
```bash
cd model
python train_yolo.py
```

**Configuration Options** (edit `train.py` or `model/train_yolo.py`):
- `epochs`: Number of training iterations (default: 10, recommended: 50-100)
- `imgsz`: Image size (default: 640)
- `batch`: Batch size (adjust based on your RAM/GPU, default: 8)

The trained model will be saved to:
```
results/brain_tumor_run/weights/best.pt
```

### Training U-Net Segmentation Model

```bash
cd model/segmentation
python train_unet.py
```

The trained model will be saved to:
```
model/segmentation/unet_best.pth
```

---

## 📊 Understanding the Results

### Tumor Types Detected
1. **Glioma**: Most common malignant brain tumor
2. **Meningioma**: Usually benign, grows slowly
3. **Pituitary**: Affects the pituitary gland
4. **No Tumor**: Healthy scan

### Staging Criteria
The system estimates tumor stage based on:
- **Tumor Area**: Measured in pixels
- **Image Coverage**: Percentage of scan occupied
- **Tumor Type**: Some types are more aggressive

**Stage Thresholds**:
- Stage I: < 3,000 pixels² or < 2% coverage
- Stage II: < 8,000 pixels² or < 5% coverage  
- Stage III: < 15,000 pixels² or < 10% coverage
- Stage IV: ≥ 15,000 pixels² or ≥ 10% coverage

### Confidence Scores
- **Above 70%**: High confidence, reliable detection
- **50-70%**: Moderate confidence, review recommended
- **Below 50%**: Low confidence, may need verification

---

## 🛠️ Troubleshooting

### Issue: "Model not found" error
**Solution**: Ensure trained model files exist:
```bash
ls results/brain_tumor_run/weights/best.pt
ls model/segmentation/unet_best.pth
```
If missing, train the models (see Training section).

### Issue: "ImportError: No module named..."
**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Slow performance
**Solutions**:
- Reduce image size (resize before upload)
- Lower the batch size during training
- Close other applications
- Consider using GPU if available

### Issue: PyTorch/CUDA errors
**Solution**: Install CPU version explicitly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Segmentation not working
**Check**:
1. U-Net model file exists: `model/segmentation/unet_best.pth`
2. PyTorch is installed: `pip install torch torchvision`
3. Check app console for error messages

### Issue: Grad-CAM not showing
**Diagnosis**: Check the sidebar status indicator
- ✅ ACTIVE: Working properly
- ❌ INACTIVE: U-Net model not loaded, using YOLO fallback

---

## 📁 Project Structure

```
Brain_tumor_detection/
├── app/
│   └── app.py                    # Main Streamlit application
├── model/
│   ├── train_yolo.py            # YOLO training script
│   ├── detect_yolo.py           # YOLO inference script
│   └── segmentation/
│       ├── unet_model.py        # U-Net architecture
│       ├── train_unet.py        # U-Net training
│       ├── predict_unet.py      # U-Net inference
│       ├── grad_cam_unet.py     # Grad-CAM visualization
│       └── unet_best.pth        # Trained U-Net weights
├── data/
│   ├── data.yaml                # YOLO dataset configuration
│   └── yolo_dataset/            # Training data
│       ├── images/
│       └── labels/
├── results/
│   └── brain_tumor_run/
│       └── weights/
│           └── best.pt          # Trained YOLO model
├── requirements.txt              # Python dependencies
├── train.py                     # Main training script
└── README.md                    # Project overview
```

---

## 🔐 Security & Privacy

- All processing happens locally on your machine
- No data is sent to external servers
- Medical images remain private
- Models are trained on anonymized public datasets

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This is an AI-assisted research and educational tool. It is **NOT** a medical device and should **NOT** be used for:
- Final medical diagnosis
- Treatment decisions
- Patient care without professional review

**Always consult qualified healthcare professionals for medical decisions.**

---

## 📞 Support & Contribution

### Getting Help
- Check this guide first
- Review the Troubleshooting section
- Check the GitHub Issues page

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with description

---

## 🎓 Technical Details

### Detection Model (YOLOv8)
- Architecture: YOLOv8n (nano version for speed)
- Input size: 640x640 pixels
- Classes: 4 (glioma, meningioma, pituitary, no_tumor)
- Framework: Ultralytics

### Segmentation Model (U-Net)
- Architecture: Simplified U-Net
- Input size: 256x256 pixels
- Output: Binary mask (tumor vs. background)
- Framework: PyTorch

### Grad-CAM
- Technique: Gradient-weighted Class Activation Mapping
- Purpose: Visual explanation of model decisions
- Layer: First encoder layer (enc1)

---

## 📚 References & Dataset

- **YOLOv8**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- **Dataset**: Brain tumor MRI dataset from Roboflow
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"

---

## 📝 Version History

### Current Version (v2.0)
- ✅ Tumor detection with YOLOv8
- ✅ Enhanced tumor staging (4 stages)
- ✅ U-Net segmentation
- ✅ Grad-CAM visualization
- ✅ Multi-criteria staging algorithm
- ✅ Web interface with Streamlit

### Features
1. **Detects tumor area** - Bounding boxes with confidence scores
2. **Stages tumors** - 4-stage classification based on size and coverage
3. **Segmentation** - Precise tumor boundary highlighting
4. **Grad-CAM** - Attention heatmaps showing AI focus areas

---

## 🎯 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files present (check with `ls` commands)
- [ ] Application launches successfully
- [ ] Can upload and analyze MRI images
- [ ] All three visualization tabs work
- [ ] Tumor staging information displays correctly

---

**Ready to use! Upload an MRI scan and see the AI in action! 🧠✨**
