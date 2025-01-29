# Breast Cancer Prediction with AutoML

This repository contains a machine learning project for classifying breast cancer cases using the Breast Cancer Wisconsin Diagnostic dataset. The project implements AutoML models, applies dimensionality reduction techniques (PCA and t-SNE), and evaluates the performance of various classifiers.

## Project Overview

The goal of this project is to predict the malignancy of breast cancer (Malignant or Benign) based on various features from cell samples. The project uses the Breast Cancer Wisconsin Diagnostic dataset, which includes a set of features extracted from digitized images of a breast mass. The project aims to automate the machine learning pipeline, improve prediction accuracy, and visualize data.

### Key Steps:
1. Load and preprocess the dataset
2. Apply PCA (Principal Component Analysis) and t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction and visualization
3. Train a classification model using AutoML
4. Evaluate the model's performance using various metrics
5. Visualize results using plots such as confusion matrix, ROC curve, and feature importance

## Features

- **Dimensionality Reduction**: 
  - PCA (Principal Component Analysis) is used to reduce the dimensionality of the dataset while preserving variance.
  - t-SNE (t-distributed Stochastic Neighbor Embedding) is used for visualizing high-dimensional data in two dimensions.
  
- **AutoML**: 
  - An AutoML pipeline is implemented to automate the machine learning model selection, training, and hyperparameter tuning. Tools like `TPOT` or `AutoSklearn` can be used for this purpose.

- **Performance Evaluation**: 
  - The model's performance is evaluated using multiple metrics:
    - **ROC AUC**
    - **Precision**
    - **Recall**
    - **F1-Score**
    - **Confusion Matrix**
  
- **Visualization**: 
  - Visualize dimensionality reduction using scatter plots from PCA and t-SNE.
  - Plot ROC Curve to evaluate model performance.
  - Display confusion matrix to understand classification results.
  - Permutation importance for feature evaluation.

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic)** dataset. The features include various measurements of cell nuclei in breast cancer biopsies, and the target variable is whether the tumor is malignant or benign.

### Data Columns:
1. **ID**: Identifier for each sample
2. **Diagnosis**: The diagnosis result (M = Malignant, B = Benign)
3. **Feature_1** to **Feature_30**: Various features like radius, texture, smoothness, etc.

## Setup and Installation

### Prerequisites

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `TPOT` (or other AutoML tools)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AshutoshJha-007/Breast-Cancer-Prediction-AutoML.git
