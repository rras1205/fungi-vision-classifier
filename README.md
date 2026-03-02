# fungi-vision-classifier

## Project Overview
* This project implements a Convolutional Neural Network (CNN) using PyTorch to classify microscopic fungi images into multiple categories.
* The model is trained on the Defungi dataset (downloaded via Kaggle) and performs supervised multi-class image classification.

This project demonstrates:
* Deep learning fundamentals
* Custom CNN architecture design
* Data preprocessing for image tasks
* Training/validation workflow
* Model evaluation using classification metrics
* Visualization of learning curves

## Dataset
Due to size constraints (158MB), the dataset is not included in this repository.

You can download it from:

https://www.kaggle.com/datasets/joebeachcapital/defungi

- Dataset used: Defungi (Microscopic Fungi Image Dataset).
- Downloaded via kagglehub.


### Classes Used

The dataset contains 5 fungal classes:

* H1
* H2
* H3
* H5
* H6

Images are .jpg files organized by class folders.

## Technologies & Libraries

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Matplotlib
* NumPy
* KaggleHub
* tqdm

## Project Pipeline 

### Dataset Download
The dataset is downloaded programmatically using:

- kagglehub.dataset_download("joebeachcapital/defungi")

Kaggle credentials are configured inside the notebook.

### Dataset Splitting
The dataset is manually split into:
- 80% Training
- 20% Testing

Images are copied into:
- /root/fungi_split/train
- /root/fungi_split/test

Each class is shuffled before splitting to avoid ordering bias.

### Image Preprocessing 
All images undergo the following transformations:
* Resize to 128 × 128
* Convert to tensor
* Normalize to range [-1, 1]

- transforms.Resize((128, 128))
- transforms.ToTensor()
- transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

Batch size: 32


## CNN Architecture

A custom CNN is implemented using nn.Sequential.

### Architecture:
1. Conv2D (3 → 32) + BatchNorm + ReLU + MaxPool
2. Conv2D (32 → 64) + BatchNorm + ReLU + MaxPool
3. Conv2D (64 → 128) + BatchNorm + ReLU + MaxPool
4. Adaptive Average Pooling
5. Fully Connected Layer (128 → 128)
6. Dropout (0.5)
7. Output Layer (128 → num_classes)

#### Key characteristics:
* Uses Batch Normalization
* Uses Dropout for regularization
* Adaptive pooling ensures fixed feature size
* Designed for small-to-medium scale image classification

## Training Configuration
* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: 0.001
* Epochs: 15
* Device Automatically selects GPU if available

Training loop tracks:
* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

## Model Evaluation
After training, the model is evaluated using:
* Confusion Matrix
* Classification Report
  * Precision
  * Recall
  * F1-score
  * Accuracy

Example evaluation tools used:

- confusion_matrix()
- classification_report()

## Training Visualization
Two learning curves are plotted:
* Loss Curve (Train vs Validation)
* Accuracy Curve (Train vs Validation)

These help assess:
* Overfitting
* Underfitting 
* Convergence behavior

## How to Run This Project

### Clone repository 
- git clone https://github.com/rras1205/fungi-vision-classifier.git

### Install Requirements 
- pip install torch torchvision scikit-learn matplotlib kagglehub tqdm

### Add Kaggle API Credentials
- Place your kaggle.json file in:

~/.kaggle/


Then run the notebook

## What this project demonstrates
This project shows:
- Ability to build a CNN from scratch
- Understanding image preprocessing
- Proper training-validation workflow
- Model evaluation beyond just accuracy
- Basic deep learning experimentation

## Limitations
This implementation:
- Uses a simple custom CNN (not pretrained models like ResNet)
- Does not use data augmentation
- Does not perform hyperparameter tuning
- Uses a simple train/test split instead of cross-validation

## Potential Improvements
- Add data augmentation (RandomRotation, RandomHorizontalFlip)
- Implement transfer learning (e.g., ResNet18)
- Add early stopping
- Save and load trained models
- Deploy as a Streamlit web app







