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

![image](https://github.com/user-attachments/assets/dd8caaf8-d1ff-499b-9501-c3d7fd1aef26)
![image](https://github.com/user-attachments/assets/fa4b6a91-a33a-4998-a3ed-a448c12768b9) 
![image](https://github.com/user-attachments/assets/2a734b1e-0d7c-4f43-b78d-8c2d782c0839)
![image](https://github.com/user-attachments/assets/7ed97a11-4db2-45f6-b4aa-61a62b7baa91)
![image](https://github.com/user-attachments/assets/49fd3c28-c58a-4d7b-a1cf-12707036cf4e)
![image](https://github.com/user-attachments/assets/8927242a-06e8-4018-b7e1-08db4270ee93)


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

![image](https://github.com/user-attachments/assets/a690a95c-d2f7-4232-94e2-3678c1642d97)

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
