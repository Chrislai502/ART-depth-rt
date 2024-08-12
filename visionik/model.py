import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionIKModel(nn.Module):
    def __init__(self, kernel_size=3):
        super(VisionIKModel, self).__init__()

        # Define the convolutional layers in a list
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        # Define Fully Connected Layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(256 * 3 * 4, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 14)  # 14-dimensional output
        ])
    
    def forward(self, x):
        # Apply all convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Apply all fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        return x