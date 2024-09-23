# model
import torch
from torch import nn

class MNISTModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(MNISTModel, self).__init__()
        
        # First input image chanel (black & white), 6 output channels, 
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, # In this case as we are only going to use the MNIST database we could change this to 1, but for convention we are going to leave this
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, # We use the same values for in and out channels to preserve the same depth to extract more detailed information from the same number of feature maps
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # We multiply 7*7 because we used two times MaxPool2d wich halves the image dimensions twice 28x28 -> 14x14 and the other max 14x14 -> 8x8
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        #print(f"Before applying the first block: {x.shape}")
        x = self.block_1(x)
        #print(f"After applying the first block: {x.shape}")
        x = self.block_2(x)
        #print(f"After applying the second block: {x.shape}")
        x = self.classifier(x)
        #print(f"After applying the classifier: {x.shape}")
        return x
    
'''
torch.manual_seed(42)
model_2 = MNISTModel(input_shape=1, 
hidden_units=10, 
output_shape=len(class_names)).to(device)
'''