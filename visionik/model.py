import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class VisionIKModel(nn.Module):
    def __init__(self, kernel_size=3):
        super(VisionIKModel, self).__init__()

        # # Load the pre-trained ShuffleNet_V2_X0_5 model with specific weights
        # self.weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        # self.model = models.shufflenet_v2_x0_5(weights=self.weights)
        # self.preprocess = self.weights.transforms()

        # # Try Mobile Net Large and Small
        # self.weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        # self.model = models.mobilenet_v3_large(weights=self.weights)
        # self.preprocess = self.weights.transforms()

        # self.weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        # self.model = models.mobilenet_v3_small(weights=self.weights)
        # self.preprocess = self.weights.transforms()

        # Define the convolutional layers inspired by ShuffleNetV2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(768, 512)
        # self.fc1 = nn.Linear(1000, 512) # For Pre-trained Weights
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 14)  # 14-dimensional output

        self.preprocess_img = transforms.Compose([
            transforms.Resize((48, 64)),
            transforms.ToTensor()   
        ])


    def forward(self, x):
        # x = self.preprocess(x)
        # x = self.model(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)        

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x