# Neural Network Practice Projects
Welcome to the Neural Network Practice Projects repository! This repository is designed to provide simple yet effective hands-on exercises to get comfortable with building and training neural networks using PyTorch.

## Overview
This repository contains a collection of projects focused on the implementation of neural networks for various datasets and tasks.

Each project is designed to teach different core concepts of neural networks, from simple feedforward networks to convolutional neural networks (CNNs). By working through these examples, gaining an understanding of how neural networks are constructed, trained, and evaluated.

## Objectives
The goal of this repository is to practice on these different milestones:

- Gain hands-on experience with PyTorch.
- Learn how to load, preprocess, and normalize different datasets.
- Understand the structure of common neural network architectures such as feedforward networks, CNNs, and more.
- Train models using different optimization algorithms like SGD and Adam.
- Visualize the training process and track performance using tools like TensorBoard.
- Get a feel for tuning hyperparameters like learning rates, batch sizes, and epoch numbers.
## Datasets Used
Throughout this repository, the following datasets will be used:

- **MNIST:** A dataset of handwritten digits (0-9) used for image classification.
- **FashionMNIST:** A dataset consisting of grayscale images of clothing items.
- **CIFAR-10:** A dataset with 32x32 color images across 10 classes (airplane, automobile, bird, cat, etc.).

## Projects
**1. Simple Feedforward Neural Network on MNIST**
This project introduces the basic building blocks of a feedforward neural network:

- Load the MNIST dataset.
- Build a fully connected neural network.
- Train the model using SGD optimizer.
- Measure performance on test data.
  
**2. Convolutional Neural Network (CNN) on CIFAR-10**
Explore the power of convolutional layers with CIFAR-10, a more complex dataset:

- Loading and normalizing CIFAR-10 images.
- Building a CNN architecture with Conv2D and MaxPooling layers.
- Training the model using Adam optimizer.
- Visualizing training progress with TensorBoard.
  
**3. Deep CNN on FashionMNIST**
Go deeper into CNNs by using more convolutional layers to achieve higher accuracy:

- Understanding overfitting and how to combat it with techniques like dropout.
- Implementing deeper architectures.
- Tracking training vs. validation loss.
  
**4. Custom Dataset and Model Training (Coming Soon)**
