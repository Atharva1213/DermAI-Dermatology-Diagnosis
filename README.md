# Derm-AI: Automated Skin Disease Diagnosis System

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Chatbot Integration](#chatbot-integration)
- [Screenshots](#screenshots)
- [Usage](#usage)
- [Results](#results)
- [Contribution Guidelines](#contribution-guidelines)

## Overview

**Derm-AI** is an AI-powered dermatological diagnosis system designed to assist users in identifying various skin diseases using deep learning models. This system is built using Convolutional Neural Networks (CNN), MobileNet, and Support Vector Machine (SVM) models. It is trained on the HAM10000 dataset and offers a user-friendly chatbot for seamless interaction.

## Features

- **Automated Skin Disease Diagnosis:** Uses deep learning models to classify skin diseases accurately.
- **Multi-Model Implementation:** Integrates CNN, MobileNet, and SVM for improved accuracy.
- **Interactive Chatbot:** Helps users upload images and get diagnosis predictions in real-time.
- **Validation and Performance Metrics:** Provides accuracy reports with validation images.
- **User-Friendly Interface:** Designed for ease of use by both medical professionals and the general public.

## Dataset

The **HAM10000 dataset** is used for training and evaluating the model. It contains **10,015** labeled images of various skin lesions, including:

- Melanocytic nevi
- Melanoma
- Benign keratosis-like lesions
- Basal cell carcinoma
- Actinic keratoses
- Vascular lesions

## Model Architecture

The system follows a **multi-stage model approach**:

1. **CNN-Based Feature Extraction:**
   - Uses a Convolutional Neural Network (CNN) to extract key features from skin images.
2. **MobileNet for Efficient Processing:**
   - MobileNet, a lightweight deep learning model, is fine-tuned for accurate classification.
3. **SVM for Final Classification:**
   - The Support Vector Machine (SVM) is used to refine the classification results for better performance.

## Implementation Details

- **Frameworks Used:** TensorFlow, Keras, OpenCV, Scikit-learn
- **Preprocessing Techniques:** Image augmentation, normalization
- **Training Strategy:**
  - Split dataset into **80% training** and **20% testing**
  - Optimized using Adam optimizer with a learning rate of 0.0001
  - Used cross-validation to prevent overfitting

## Chatbot Integration

We implemented a chatbot that:

- **Allows Users to Upload Images:** Users can upload images via the chatbot interface.
- **Provides Real-Time Predictions:** The chatbot processes the image and returns diagnosis results.
- **Gives General Advice:** Offers insights into common skin conditions and next steps.
- **Built Using:** Dialogflow, Flask API, and React for the frontend.

## Screenshots

![image](https://github.com/user-attachments/assets/04a2656a-8400-4268-9d57-fdbd5881e52d) 
![image](https://github.com/user-attachments/assets/1811a9e4-2f0e-4f67-baa5-dd6b2c6c88f1) 
![image](https://github.com/user-attachments/assets/e212d8dd-1ec0-454d-baf7-475696c315c0) 
![image](https://github.com/user-attachments/assets/befb68db-fafe-43d0-b1ca-0a50bff77be9) 
![image](https://github.com/user-attachments/assets/11b60e2c-67ed-4bde-a1a3-c41f7cc6cdc7) 
![image](https://github.com/user-attachments/assets/1994aee1-55f8-4807-825e-e8779a8f8f82)


## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Atharva1213/DermAI-Dermatology-Diagnosis.git
cd DermAI-Dermatology-Diagnosis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

## Results

The model was evaluated on a test set from the **HAM10000 dataset**, achieving:

- **CNN Accuracy:** 85%
- **MobileNet Accuracy:** 99.2%
- **SVM Accuracy:** 96%

### Validation Images

![image](https://github.com/user-attachments/assets/143f9b2a-9f43-420f-a624-4676dde56776)

## Contribution Guidelines

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request for review.
