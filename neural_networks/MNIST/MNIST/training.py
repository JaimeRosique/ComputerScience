# training.py
import torch
from torch import nn
from model import MNISTModel
from data_preparation import train_loader, test_loader
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter

LEARNING_RATE = 0.001
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
total_acc = 0.0

model = MNISTModel(input_shape=1,hidden_units=10,output_shape=10).to(DEVICE)  # Send the model to the device (CPU/GPU)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(model, train_loader, criterion, optimizer):
    global total_acc
    best_vloss = 1_000_000. # Large initial value for best validation cost
    
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}: ")

        # Make sure gradient tracking is on
        model.train(True)
        running_loss = 0.0
        log_interval = len(train_loader) // 10  # Log every 10% of the total batches
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            
            # To zero gradients for every batch 
            optimizer.zero_grad()
            
            # Forward pass: Make predictions for this batch
            outputs = model(inputs)
            
            # Compute loss and gradients (backprop)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()
            
            # Gather data and report 
            running_loss += loss.item()
            if (i + 1) % log_interval == 0:
                avg_loss = running_loss/1000 # Calcula la perdida promedio por cada 1000 lotes
                print(f"batch: {i+1} loss: {avg_loss}")
                #writer.add_scalar('Loss/train', avg_loss, epoch * len(train_loader) + i + 1)
                running_loss = 0.0
                
        # Validation phase
        model.eval() # set the model to evaluation mode
        running_vloss = 0.0
        total = 0.0
        correct = 0.0
        
        # Disable gradient computation to save memory and computation
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
                 # Calcular precisión
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = 100 * correct / total
        total_acc += accuracy
        print(f"Test loss: {running_loss/len(test_loader)}, Accuracy: {accuracy}%")
            
        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss:.4f}, valid {avg_vloss:.4f}')    
        
        # Log the running loss averaged per batch for training and validation
        #writer.add_scalars('Loss/train vs. Validation', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)

        # Track the best validation loss and save the model state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved at epoch {epoch + 1}')

train(model, train_loader, criterion, optimizer)