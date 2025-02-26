# Transfer Learning for Image Classification

## Overview

This project explores **transfer learning** for **image classification**, leveraging **deep learning** models trained on large datasets (such as ImageNet) to classify images into six scene categories. Instead of training a deep network from scratch, we fine-tune pre-trained networks—**ResNet50, ResNet101, EfficientNetB0, and VGG16**—to achieve high classification accuracy.

## Motivation

Deep learning models often require vast amounts of labeled data to generalize well. **Transfer learning** helps mitigate this challenge by utilizing pre-trained models that have already learned hierarchical image features. In this project, we extract features from pre-trained networks and train a new **fully connected layer** on top to classify scene images.

## Key Features

- **Data Preprocessing & Augmentation**
  - One-hot encoding for multi-class classification.
  - Image resizing and zero-padding to ensure uniform dimensions.
  - **Augmentations**: cropping, zooming, rotation, flipping, contrast adjustment, and translation (using OpenCV).

- **Transfer Learning Implementation**
  - Use **pre-trained models** (ResNet50, ResNet101, EfficientNetB0, VGG16).
  - **Freeze** all layers except the last fully connected layer.
  - Train a **new classification head** with a **softmax activation function**.

- **Model Training & Optimization**
  - **Regularization**: L2 weight decay, dropout (20%), batch normalization.
  - **Optimizer**: Adam
  - **Loss function**: Multinomial cross-entropy
  - **Early Stopping**: Stop training based on validation loss.

- **Performance Evaluation**
  - Metrics: **Precision, Recall, AUC, and F1 Score**
  - Comparison of **training, validation, and test performance** across models.
  - **Visualization** of training curves: loss and accuracy vs. epochs.

## Dataset

The dataset consists of **six scene categories**, with separate **training and test sets**. The images were resized and preprocessed before being used for training.

## Results

- Compared **four deep learning models** on a multi-class classification problem.
- **Achieved high accuracy** using **pre-trained networks** with minimal training.
- **EfficientNetB0 and ResNet50** showed the best generalization on test data.

## Technologies Used

- **Python** (NumPy, Pandas, Matplotlib)
- **Deep Learning**: Keras, TensorFlow
- **Computer Vision**: OpenCV
- **Model Evaluation**: Precision, Recall, AUC, F1 Score

## Future Improvements

- Fine-tune additional layers beyond the last fully connected layer.
- Experiment with **different dropout rates and batch sizes**.
- Use **semi-supervised learning** to further enhance performance.

## Author

This project was developed as part of **DSCI 552** at the **University of Southern California (USC)**. It serves as a demonstration of deep learning skills, model evaluation, and data science best practices.

## License

This project is open-source and available under the **MIT License**.


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/RuHimdEP)

# Multi-Class-Image-Classification-Using-Transfer-Learning-
Transfer learning-based image classification project using pre-trained deep learning models (ResNet50, ResNet101, EfficientNetB0, VGG16). Implements fine-tuning, image augmentation, and model evaluation for scene classification.
>>>>>>> origin/main
