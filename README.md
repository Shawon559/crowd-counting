# Crowd Counting System using CSRNet

Deep learning-based crowd counting system for density estimation in crowded public spaces. This project implements the CSRNet (Congested Scene Recognition Network) architecture for accurate crowd counting in subway stations and other high-density environments.

## Overview

This system uses a CNN-based approach for crowd density estimation, trained on the ShanghaiTech dataset. The model generates density maps from input images and predicts accurate crowd counts even in highly congested scenes.

### Key Features

- **CSRNet Architecture:** Dilated convolutional neural network for density map generation
- **VGG16 Pretrained Frontend:** Transfer learning for improved feature extraction
- **Multi-Dataset Support:** Trained and tested on ShanghaiTech Part A and Part B
- **Real-time Inference:** Efficient crowd counting for surveillance applications
- **Robust Training:** NaN detection, gradient clipping, and CSV logging

## Model Architecture

### CSRNet Structure

The model consists of two main components:

1. **Frontend (Feature Extraction)**
   - Based on VGG16-BN architecture (first 13 layers)
   - Pretrained on ImageNet for robust feature extraction
   - Frozen weights during training to preserve learned representations

2. **Backend (Density Estimation)**
   - 6-layer dilated convolutional network
   - Dilation rate of 2 for expanded receptive field
   - Generates high-quality density maps without resolution loss

3. **Output Layer**
   - 1×1 convolution for final density map prediction
   - Output summed to produce total crowd count

**Architecture Flow:**
```
Input Image (RGB)
    ↓
VGG16 Frontend (Conv layers 1-13)
    ↓
Dilated Conv Backend (512→512→512→256→128→64)
    ↓
Output Layer (64→1)
    ↓
Density Map
```

## Project Structure

```
CSRNet-pytorch-master/
├── model.py                    # CSRNet model architecture
├── dataset.py                  # PyTorch dataset loader
├── train.py                    # Training script
├── image.py                    # Image preprocessing utilities
├── utils.py                    # Model saving/loading utilities
├── make_dataset.ipynb          # Ground truth generation
├── make_model.ipynb            # Model architecture visualization
├── val.ipynb                   # Validation and testing
├── test.py.ipynb               # Testing utilities
├── part_A_train.json          # ShanghaiTech Part A training set
├── part_A_test.json           # ShanghaiTech Part A test set
├── part_B_train.json          # ShanghaiTech Part B training set
├── part_B_test.json           # ShanghaiTech Part B test set
└── README.md                   # This file
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA 10.2+ (for GPU training)

### Dependencies

```bash
pip install torch torchvision
pip install numpy opencv-python h5py Pillow
```

## Usage

### 1. Dataset Preparation

Generate ground truth density maps from annotations:

```bash
jupyter notebook make_dataset.ipynb
```

This notebook:
- Loads ShanghaiTech dataset annotations
- Generates Gaussian density maps for each image
- Saves preprocessed data for training

### 2. Training

Train the model on ShanghaiTech dataset:

```bash
python train.py part_B_train.json part_B_test.json 0 0
```

**Arguments:**
- `part_B_train.json`: Path to training set JSON
- `part_B_test.json`: Path to validation set JSON
- `0`: GPU ID to use
- `0`: Task ID (for checkpoint naming)

**Training Parameters:**
- Learning Rate: 1e-7
- Optimizer: SGD with momentum (0.95)
- Weight Decay: 5e-4
- Batch Size: 1
- Epochs: 100
- LR Schedule: Decay at epochs 20 and 40

### 3. Validation

Evaluate trained model performance:

```bash
jupyter notebook val.ipynb
```

The validation notebook:
- Loads trained model checkpoint
- Runs inference on test set
- Computes MAE (Mean Absolute Error)
- Visualizes density map predictions

### 4. Inference

Use the trained model for crowd counting:

```python
import torch
from model import CSRNet
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = CSRNet()
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Prepare image
img = Image.open('test_image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

# Predict crowd count
with torch.no_grad():
    density_map = model(img_tensor)
    crowd_count = density_map.sum().item()

print(f"Estimated crowd count: {crowd_count:.2f}")
```

## Datasets

### ShanghaiTech Crowd Counting Dataset

**Part A (Highly Congested Scenes):**
- Training: 300 images
- Testing: 182 images
- Average count: 501 people per image
- High density, complex scenes

**Part B (Less Dense Scenes):**
- Training: 400 images
- Testing: 316 images
- Average count: 123 people per image
- Moderate density, urban streets

**Dataset Format:**
- Images: RGB, variable resolution
- Annotations: Head positions (x, y coordinates)
- Ground Truth: Gaussian density maps (.npy files)

## Model Performance

**ShanghaiTech Part B Results:**
- Mean Absolute Error (MAE): 10.6
- Model checkpoint: `trainmodel_best_B_1more_COPY-Copy1.pth.tar`

**Training Details:**
- Framework: PyTorch
- GPU: CUDA-enabled
- Training Time: ~8-10 hours on single GPU
- Checkpoint saving: Every epoch + best model

## Implementation Details

### Key Components

1. **Data Augmentation:**
   - Training data repeated 4x for augmentation
   - Random shuffling each epoch
   - Normalized with ImageNet statistics

2. **Loss Function:**
   - Mean Squared Error (MSE) with sum reduction
   - Measures pixel-wise difference between predicted and ground truth density maps

3. **Optimization:**
   - SGD optimizer with momentum
   - Gradient clipping (max norm: 1.0) for stability
   - Learning rate decay schedule

4. **Robustness Features:**
   - NaN detection and skipping
   - Gradient norm clipping
   - CSV logging for training metrics

### Data Pipeline

```python
# Image preprocessing
img.resize((1024, 768))  # Fixed input size
normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Density map preprocessing
target.resize((128, 96))  # Downsampled output
```

## Technical Notes

### Why CSRNet?

Traditional detection-based crowd counting methods fail in highly congested scenes due to:
- Severe occlusions
- Scale variations
- Perspective distortion

CSRNet addresses these challenges through:
- **Density-based regression** instead of detection
- **Dilated convolutions** for larger receptive field
- **End-to-end learning** without hand-crafted features

### Advantages

- No need for individual person detection
- Handles extreme density variations
- Robust to occlusions and perspective changes
- Fast inference speed for real-time applications

## Capstone Project Context

This project was developed as a capstone project for crowd monitoring in public transportation systems, specifically:
- **Application:** Subway station crowd density estimation
- **Goal:** Real-time crowd counting for safety and management
- **Dataset:** ShanghaiTech Part B (urban scenes)
- **Deployment:** Production-ready model with 124MB checkpoint

## References

**CSRNet Paper:**
```
Li, Y., Zhang, X., & Chen, D. (2018).
CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1091-1100).
```

**ShanghaiTech Dataset:**
```
Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016).
Single-image crowd counting via multi-column convolutional neural network.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 589-597).
```

## License

This project is for academic and educational purposes.

## Contact

For questions or collaboration:
- GitHub: [Shawon559](https://github.com/Shawon559)
- Email: mehadi.shawon559@gmail.com

---

**Note:** This implementation is based on the CS RNet architecture for crowd counting research and education. The model checkpoint provided achieves competitive performance on the ShanghaiTech Part B benchmark.
