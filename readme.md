# CNN vs ResNet18: Understanding Residual Learning through Image Classification

## Overview

This project was developed while studying the paper **"Deep Residual Learning for Image Recognition"** by Kaiming He et al.

The primary goal of this project is to understand the practical impact of **Residual Learning** by comparing a traditional Convolutional Neural Network (CNN) with a Residual Network (ResNet18) on the CIFAR-10 image classification dataset.

To further explore the models, an interactive Streamlit application was developed that allows users to upload images and compare the predictions made by both models.

---

## Motivation

While reading the ResNet paper, I was particularly interested in the degradation problem observed in deep neural networks. The authors showed that simply increasing network depth can lead to higher training error, despite deeper networks having greater representational power.

ResNet addresses this issue through **Residual Learning** and **Skip Connections**, allowing networks to learn residual mappings instead of complete transformations.

This project was developed to investigate these ideas experimentally.

---

## Dataset

The models were trained on the CIFAR-10 dataset.

### Classes

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset Statistics:

* Training Images: 50,000
* Test Images: 10,000
* Number of Classes: 10

---

## Models Implemented

### 1. Traditional CNN

Architecture:

Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Fully Connected Layers → Output

This model serves as a baseline for comparison.

### 2. ResNet18

ResNet18 utilizes residual blocks and shortcut connections.

Residual Block:

Output = F(x) + x

where:

* x is the input
* F(x) is the residual mapping learned by the network

The model was adapted for CIFAR-10 classification by replacing the final classification layer with a 10-class output layer.

---

## Experimental Setup

* Framework: PyTorch
* Dataset: CIFAR-10
* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Training Environment: Google Colab (GPU)
* Deployment Environment: Local CPU / Streamlit

---

## Results

### Training Loss Comparison

The training loss of both CNN and ResNet18 was monitored across multiple epochs.

[Insert Loss vs Epoch Graph]

### Accuracy Comparison

| Model    | Accuracy |
| -------- | -------- |
| CNN      | 62.10%   |
| ResNet18 | 90.45%   |

### Observations

* ResNet18 achieved higher classification accuracy than the traditional CNN.
* ResNet18 converged more efficiently during training.
* The results support the central claim of the ResNet paper that residual learning improves optimization and enables more effective training of deep neural networks.

---

## Interactive Application

A Streamlit-based web application was developed to compare the predictions of both models.

### Features

* Upload an image
* Display image preview
* Generate CNN prediction
* Generate ResNet18 prediction
* Compare model outputs

Example:

Input Image: Dog

CNN Prediction:
Dog

ResNet18 Prediction:
Dog

---

## Limitations

The models were trained exclusively on the CIFAR-10 dataset and can classify images into only the following ten categories:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Images belonging to other categories will still be assigned to the closest available class.

---

## Key Learning

The most important lesson from this project was understanding that the success of ResNet is not merely due to increased depth, but due to its ability to simplify optimization through residual learning.

The experiments reinforced the paper's insight that learning residual mappings is often easier than learning complete transformations, enabling the successful training of much deeper neural networks.

---

## Future Work

* Compare deeper ResNet variants (ResNet34, ResNet50)
* Evaluate performance on larger datasets
* Visualize feature maps and learned representations
* Explore other architectures that utilize residual connections (e.g., DenseNet, ResNeXt)

---

## References

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

**Deep Residual Learning for Image Recognition**

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
