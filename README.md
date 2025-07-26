# Brain Tumor Classifier

A deep learning-based web application that detects brain tumors from MRI images.  
The model classifies scans into one of four categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, or **No Tumor**.  
Built with **TensorFlow** for model training and **Streamlit** for deployment, the app provides a simple interface to upload images and receive instant predictions.

---

## Brief Description

This project presents an intelligent system for automated brain tumor classification using MRI images. Leveraging a machine learning model trained on a diverse dataset (integrating sources like Figshare, SARTAJ, and Br35H), the system classifies brain tumors into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**. The image preprocessing pipeline includes grayscale conversion, Gaussian blurring, and ROI cropping to ensure high-quality input data.

The trained model is deployed via a Streamlit web application, allowing users to upload MRI images and receive real-time predictions. This tool streamlines the diagnostic process, reduces human error, and assists medical professionals in early and accurate tumor detection. Designed to be simple, accessible, and efficient, the project aims to support improved patient outcomes and smarter healthcare resource utilization.

---

## Live Demo

Try the web app without installing anything:  
[https://deployedbtc.streamlit.app](https://deployedbtc.streamlit.app)

---

## Demo

![Demo](https://github.com/nameeshsachdev2025/Brain_Tumor_classification/blob/main/brain_tumor_demo2.gif?raw=true)

---

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/nameeshsachdev2025/Brain_Tumor_classification.git
cd Brain_Tumor_classification
pip install -r requirements.txt
```

## Dataset Overview

The dataset used for training and evaluating the brain tumor classifier contains MRI scan images divided into four classes. All images have been resized to 150×150×3 and organized into separate folders for training, validation, and testing.

### Classes
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

### Dataset Structure

```txt
Dataset/
├── Train/
│   ├── Glioma Tumor/
│   ├── Meningioma Tumor/
│   ├── Pituitary Tumor/
│   └── No Tumor/
│
├── Test/
│   ├── Glioma Tumor/
│   ├── Meningioma Tumor/
│   ├── Pituitary Tumor/
│   └── No Tumor/
│
└── Validation/
    ├── Glioma Tumor/
    ├── Meningioma Tumor/
    ├── Pituitary Tumor/
    └── No Tumor/
```


Each subfolder contains `.jpg` or `.png` MRI images labeled according to tumor type.

---

### Dataset Summary

- **Image Size:** 150 × 150 × 3 (RGB)
- **Training Samples:** 5,721
- **Validation Samples:** 1,311
- **Test Samples:** ~1,600 (approx.)

---
## Model Architecture

The model is built using a Convolutional Neural Network (CNN) designed to classify brain MRI images into four categories. The architecture includes:

- **Convolutional Layers** with ReLU activation to learn spatial features
- **MaxPooling Layers** to reduce dimensionality
- **Dropout Layers** for regularization and overfitting control
- **Fully Connected (Dense) Layers** for final classification
- **Softmax Output Layer** to produce probability scores for 4 classes

### Transfer Learning

To enhance performance, pretrained architectures like **VGG16** and **ResNet50** were also explored. These models were fine-tuned by replacing their classification heads and training on the brain tumor dataset.

### Data Augmentation

To increase robustness and generalization, real-time data augmentation was applied using `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='reflect'
)

train_generator = datagen.flow(x_train, y_train, batch_size=32)
```
## Tech Stack

**Model & Backend:**

- Python
- TensorFlow / Keras
- NumPy, Pandas, OpenCV
- Scikit-image, Matplotlib

**Web Application:**

- Streamlit

**Deployment:**

- Streamlit Cloud

**Development Tools:**

- Jupyter Notebook
- Google Colab
- Git & GitHub

## Confusion Matrix

![Confusion Matrix](https://raw.githubusercontent.com/nameeshsachdev2025/Brain_Tumor_classification/main/Brain_Tumor_classification/confusion_matrix_final.png)


## Acknowledgments

- Dataset sources: [Figshare](https://figshare.com), SARTAJ, Br35H
- Streamlit for easy deployment
- TensorFlow & Keras for deep learning framework




