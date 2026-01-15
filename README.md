# 🧬 ISIC 2019 Skin Lesion Classification

## 📋 Project Overview

This project implements a complete deep learning pipeline for classifying skin lesion images from the ISIC 2019 dataset. The system includes both **TensorFlow** and **PyTorch** implementations with comprehensive data preprocessing, medical image engineering, and model training workflows.

## 🎯 Objectives

- Develop robust CNN models for 8-class skin lesion classification
- Implement medical-specific image preprocessing (CLAHE, hair removal)
- Address class imbalance through weighted loss functions
- Compare TensorFlow and PyTorch implementations
- Provide reproducible pipeline for medical image analysis

## 📁 Dataset

**ISIC 2019 Dataset** - Dermoscopic images with 8 diagnostic categories:

| Class | Description | Prevalence |
|-------|-------------|------------|
| MEL | Melanoma | Moderate |
| NV | Melanocytic Nevus | High |
| BCC | Basal Cell Carcinoma | Moderate |
| AK | Actinic Keratosis | Low |
| BKL | Benign Keratosis-like Lesions | Moderate |
| DF | Dermatofibroma | Low |
| VASC | Vascular Lesions | Low |
| SCC | Squamous Cell Carcinoma | Low |

## 🏗️ Project Structure

```
SkinConditionAssistant/
│
├── data/
│   ├── raw/                            # Original ISIC 2019 data
│   ├── processed/                      # Resized images (512x512)
│   ├── train/                          # Training imgs
│   ├── val/                            # Validation imgs
│   └── test/                           # Test imgs
│
├── data_cleaning/                      # Working on data and splitting for train, val and test
│   ├── clean_data.py
│   ├── EDA.py
│   ├── image_engineering.py
│   └── paths.py
│   └── train_test_split.py
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb       # Data preparation & image engineering
│   ├── 02_Train_Val_Test_Split.ipynb   # Data splitting for training
│   └── 03_Model_TensorFlow.ipynb       # TensorFlow model training
│   └── 04_Model_PyTorch.ipynb          # PyTorch model training
│
├── outputs/
│   ├── metrics/
│   ├── confusion_matrices/
│   └── saved_models/
|
├── pytorch_training/
│   └── model_training.py
|
├── tensorflow_training/
│   └── model_training.py
│
└── README.md
```

## 🔄 Data Pipeline

### 1. Data Preparation (`data_preprocessing.ipynb`)
- **Label Processing**: Extracts dominant class from one-hot encoded CSV
- **Image Resizing**: Standardizes all images to 512×512 pixels
- **Data Verification**: Integrity checks and corrupted image removal
- **Class Distribution Analysis**: Visualizes dataset imbalance

### 2. Medical Image Engineering
- **CLAHE Enhancement**: Improves local contrast in lesion regions
- **Hair Removal**: Morphological processing to reduce occlusion artifacts
- **Color Space Conversion**: Ensures consistent RGB format

### 3. Dataset Splitting
- **70% Train** / **15% Validation** / **15% Test** split
- **Stratified sampling** to maintain class distribution
- **Directory organization** compatible with both frameworks

## 🤖 Model Architectures

### TensorFlow Implementation
- **Backbone**: EfficientNetB0 (pretrained on ImageNet)
- **Fine-tuning**: First 150 layers frozen, last 50 layers trainable
- **Custom Head**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(256, ReLU)
  - Dropout(0.5)
  - Dense(9, Softmax)

### PyTorch Implementation
- **Backbone**: EfficientNet-B0 (pretrained)
- **Gradual Unfreezing**: Progressive unfreezing schedule over 80+ epochs
- **Custom Classifier Head**:
  - Dropout(0.5)
  - Linear(in_features, 256)
  - ReLU + BatchNorm1d
  - Dropout(0.5)
  - Linear(256, num_classes)

## ⚙️ Training Configuration

### Common Settings
- **Optimizer**: Adam/AdamW (lr=5e-5)
- **Loss Function**: Sparse Categorical Crossentropy / Focal Loss
- **Class Weights**: Inverse frequency weighting with sqrt scaling
- **Batch Size**: 64
- **Image Size**: 512×512 (resized to 224×224 for models)

### TensorFlow Specific
- **Callbacks**: ReduceLROnPlateau, EarlyStopping
- **Metrics**: SparseCategoricalAccuracy, Top-2 Accuracy

### PyTorch Specific
- **Mixed Precision Training**: FP16 for memory efficiency
- **Gradual Unfreezing**: 5-stage unfreezing schedule
- **DataLoader Optimizations**: Persistent workers, prefetching, pin memory

## 📊 Data Augmentation

### Training Augmentations
- Random horizontal flip (p=0.3)
- Random rotation (±3°)
- Random resized crop (scale: 0.92-1.0)
- Color jitter (brightness, contrast, saturation, hue: 0.05)
- Gaussian noise (std=0.02)

### Medical Imaging Considerations
- **Mild augmentations** to preserve diagnostic features
- **No extreme transformations** that could alter clinical interpretation
- **Consistent normalization** using ImageNet statistics

## ⚖️ Handling Class Imbalance

1. **Class Weighting**: Inverse frequency weighting
2. **Focal Loss (PyTorch)**: Focuses on hard examples
3. **Stratified Sampling**: Maintains distribution in splits
4. **Evaluation Metrics**: Precision, recall, F1-score per class

## 📈 Performance Results

### TensorFlow Model
- **Test Accuracy**: 74.68%
- **Test Loss**: 1.485
- **Best Performance**: BCC (F1: 0.86), BKL (F1: 0.74)
- **Challenging Classes**: VASC (F1: 0.38), DF (F1: 0.53)

#### Key Insights
- Strong performance on common lesions (BCC, BKL)
- Expected difficulty on rare classes (VASC, DF)
- Moderate melanoma recall (0.52) - important clinical target
- Confusion between benign-looking classes (NV, BKL)

### PyTorch Model
- **Test Accuracy**: 74.36%
- **Test Loss**: 1.285
- **Best Performance**: BCC (F1: 0.84), SCC (F1: 0.89)  
- **Challenging Classes**: VASC (F1: 0.39), DF (F1: 0.56)

#### Key Insights:
- **Strong on common lesions (BCC, BKL)** – F1: 0.84 and 0.71
- **Challenged by rare classes (VASC, DF)** – F1: 0.39 and 0.56
- **Melanoma recall moderate (0.58)** – Critical clinical target
- **Confusion among benign classes (NV, BKL)** – Overlapping visual features

## 🚀 Usage

### Prerequisites
```bash
pip install tensorflow torch torchvision pandas pillow opencv-python scikit-learn matplotlib seaborn joblib tqdm
```

### Get Dataset
Download ISIC 2019 dataset from: https://challenge.isic-archive.com/data/?utm_source=chatgpt.com#2019
and extract it in data/raw folder

### Quick Start
1. **Prepare Data**:
```python
# Run data_preprocessing.ipynb
# This will create processed/ and worked/ directories
```

2. **Train TensorFlow Model**:
```python
# Run tensorflow_training.ipynb
# Models saved to models/tensorflow/
```

3. **Train PyTorch Model**:
```python
# Run model_training.py in pytorch_training folder 
# Models saved to models/pytorch/
```

### Inference Example
```python
# TensorFlow
model = tf.keras.models.load_model('models/tensorflow/skin_condition_model.keras')
prediction = model.predict(preprocessed_image)

# PyTorch
model = EfficientNetB0PyTorch(num_classes=9)
model.load_state_dict(torch.load('models/pytorch/best_model.pth'))
prediction = model(image_tensor)
```

## 🔍 Model Interpretability

Both implementations include:
- **Confusion matrices** for error analysis
- **Classification reports** with precision/recall/F1
- **Training history visualization**
- **Class-wise performance analysis**

## 🎯 Key Features

### Medical-Specific
- **CLAHE enhancement** for better lesion visibility
- **Hair removal** to reduce occlusion artifacts
- **Domain-appropriate augmentations**
- **Clinical validity preservation**

### Engineering Excellence
- **Reproducible pipelines** with configuration management
- **Comprehensive logging** and checkpointing
- **GPU optimization** for both frameworks
- **Modular code structure** for maintainability

### Research-Ready
- **Both TensorFlow and PyTorch implementations**
- **Detailed performance metrics**
- **Visualization utilities**
- **Exportable models** for deployment

## 📝 Notes & Considerations

1. **Dataset Imbalance**: The ISIC 2019 dataset is heavily imbalanced (NV ≫ other classes)
2. **Medical Caution**: Models are for educational/research purposes only
3. **Compute Requirements**: Training requires GPU for reasonable runtime
4. **Storage**: Processed dataset requires ~2-3GB disk space

## 🔮 Future Improvements

1. **Advanced Architectures**: Vision Transformers, DenseNet
2. **Ensemble Methods**: Combine TensorFlow and PyTorch predictions
3. **Explainability**: Grad-CAM, attention visualization
4. **Multimodal Learning**: Incorporate patient metadata
5. **Deployment**: Web interface, API service

## 📚 References

1. ISIC 2019 Challenge: https://challenge.isic-archive.com/
2. EfficientNet: Tan & Le, 2019
3. Focal Loss: Lin et al., 2017
4. Medical Image Preprocessing: Tschandl et al., 2018

## 👥 Contributors

- Data preprocessing and augmentation pipeline
- TensorFlow implementation with EfficientNet
- PyTorch implementation with gradual unfreezing
- Comprehensive evaluation and visualization

## ⚖️ License

Educational/Research Use Only - Not for Clinical Diagnosis

---


*This project demonstrates a complete deep learning pipeline for medical image analysis, balancing research rigor with practical implementation considerations.*




