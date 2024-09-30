#MNIST Neural Network - 99.8% Accuracy
This repository contains my first neural network built from scratch, designed to classify handwritten digits from the MNIST dataset. The model achieves an impressive 99.8% accuracy on the test set, reflecting the knowledge and techniques I’ve acquired through self-study.

##Overview
The MNIST dataset is a benchmark dataset used for evaluating models on the task of handwritten digit recognition. It consists of 60,000 training images and 10,000 test images, each a grayscale image of 28x28 pixels representing digits from 0 to 9.

In this project, I designed and trained a neural network using deep learning techniques to classify the digits accurately. The network was implemented and trained based on concepts I’ve learned independently, and this is my first neural network project.

##Key Features
-**Model Type**: Fully connected feedforward neural network
-**Accuracy**: Achieves 99.8% accuracy on the MNIST test set
-**Dataset**: MNIST Dataset
-**Framework**: TensorFlow or PyTorch (Choose the one you used)

##Training the Model
To train the model, simply run the Python script:

python training.py

The script will:

-Load the MNIST dataset
-Build and compile the neural network model
-Train the model on the training data
-Evaluate the model on the test set

##Model Architecture
The neural network is a fully connected feedforward model with the following architecture:

-Input Layer: 784 neurons (28x28 pixels flattened)
-Hidden Layers: 2-3 hidden layers with ReLU activations
-Output Layer: 10 neurons (one for each digit 0-9) with softmax activation
**Results**
-Training Accuracy: ~99.9%
-Test Accuracy: 99.8%
The model's performance demonstrates a high level of generalization, successfully recognizing handwritten digits with near-perfect accuracy.

##Future Improvements
Although the model already achieves high accuracy, there are several potential improvements:

-**Data Augmentation**: Applying transformations (e.g., rotations, shifts) to expand the dataset.
-**Regularization Techniques**: Incorporating dropout or L2 regularization to prevent overfitting.
-**Model Tuning**: Further hyperparameter tuning (e.g., learning rate, batch size) could lead to even better results.

##Conclusion
This project represents my first venture into neural networks and deep learning. Through this experience, I’ve gained valuable insights into model training, optimization, and evaluation. Throught the next future proyects I look forward to furthering my knowledge, understanding of CNNs and RNNs and other types of NNs.
