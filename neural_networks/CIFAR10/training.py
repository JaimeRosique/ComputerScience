# training.py
import torch
from torch import nn
from model import CIFAR10Model
from data_preparation import train_loader, test_loader
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 7 #3
MOMENTUM = 0.7 #0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss function, and optimizer
model = CIFAR10Model(input_shape=3, hidden_units=128, output_shape=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

def train(model, train_loader, criterion, optimizer):
    best_vloss = float('inf')  # Use float('inf') for better readability
    for epoch in range(EPOCHS):
        logging.info(f"EPOCH {epoch + 1}: ")
        model.train()
        running_loss = 0.0
        log_interval = len(train_loader) // 10
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                logging.info(f"Batch: {i + 1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        # Validation phase
        avg_vloss, accuracy = validate(model, test_loader, criterion, best_vloss, epoch + 1)
        logging.info(f'Test accuracy at {accuracy} with validation loss: {avg_vloss:.4f}')

        # Save the model if the validation loss is the best so far
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            logging.info(f'Model saved at {model_path} with validation loss: {avg_vloss:.4f}')

def validate(model, test_loader, criterion, best_vloss, epoch):
    model.eval()
    running_vloss = 0.0
    total = 0.0
    correct = 0.0
    
    with torch.no_grad():
        for vinputs, vlabels in test_loader:
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()

            _, predicted = torch.max(voutputs, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()

    avg_vloss = running_vloss / len(test_loader)
    accuracy = 100 * correct / total
    logging.info(f"Epoch {epoch} - Test loss: {avg_vloss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_vloss, accuracy

train(model, train_loader, criterion, optimizer)
