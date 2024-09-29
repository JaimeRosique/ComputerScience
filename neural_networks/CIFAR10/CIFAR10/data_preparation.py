"""
    Prepares the CIFAR-10 dataset for training and testing using PyTorch.

    Steps:
    1. Defines transformations: converts images to tensors and normalizes them (making a precalculation to identify the mean and variance of the dataset).
    2. Downloads and loads the CIFAR-10 dataset (if not already available).
    3. Creates DataLoader objects for managing training and test data batches.

    Variables:
    - BATCH_SIZE: Number of samples per batch.
    - train_dataset: Loader of the training dataset
    - test_dataset: Loader of the test dataset
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for test data.
"""
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
train_dataset = datasets.CIFAR10(
    root='./data',  # Directory where the dataset will be saved
    train=True,     # True means training data, False means test data
    transform=transform,  # Apply transformations
    download=True,   # Download the dataset if it's not already present
    target_transform=None  # Does not transform labels
)

# Download and load the test dataset
test_dataset = datasets.CIFAR10(
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
images_loader, labels_loader = next(iter(train_loader))
print(f"Train data: {images_loader.shape}, Train labels: {labels_loader.shape}")

plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images_loader[i].permute(1, 2, 0), cmap='gray')
    plt.title(train_dataset.classes[labels_loader[i].item()])
    plt.axis("off") 
plt.show()
'''
