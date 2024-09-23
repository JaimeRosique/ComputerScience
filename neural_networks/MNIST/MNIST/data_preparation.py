# data_preparation

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
    
BATCH_SIZE = 32    
    
# Define the transformation to convert the images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and Std for normalization (precalculated)
])

# Download and load the training dataset
train_dataset = datasets.MNIST(
    root='./data',  # Directory where the dataset will be saved
    train=True,     # True means training data, False means test data
    transform=transform,  # Apply transformations
    download=True,   # Download the dataset if it's not already present
    target_transform=None  # Does not transform labels
)

# Download and load the test dataset
test_dataset = datasets.MNIST(
                root='./data',
                train=False,
                transform=transform,
                download=True
            )

# Define the DataLoader to manage data lots 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# This is to understand and check how many batches we have done 
'''
print(f"Length of train dataloader: {len(train_loader)} batch size: {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_loader)} batch size: {BATCH_SIZE}")
'''

# This is to understand the shape of the training sets and the number of classes 
'''
images, labels = train_dataset[0]

print(f"Images batch dimensions: {images.shape}") # to check dimensions and Number of classes

print(f"Num samples (train data, train targets, test data, test targets): {len(train_dataset.data), len(train_dataset.targets), len(test_dataset.data), len(test_dataset.targets)}") # how many samples there are

print(f"class names: {train_dataset.classes}")
'''

# This is to check how is the training set of images
'''
images, labels = next(iter(train_loader))
print(f"Train data: {images.shape}, Train labels: {labels.shape}")

plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(train_dataset.classes[labels[i]])
    plt.axis("off") 
plt.show()
'''