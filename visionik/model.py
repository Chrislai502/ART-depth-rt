from collections import OrderedDict
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from omegaconf import DictConfig

class VisionIKModel(nn.Module):
    def __init__(self):
        super(VisionIKModel, self).__init__()

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
    
class ShuffleNet_V2_X0_5(nn.Module):
    def __init__(self):
        super(ShuffleNet_V2_X0_5, self).__init__()

        # Load the pre-trained ShuffleNet_V2_X0_5 model with specific weights
        self.weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        self.model = models.shufflenet_v2_x0_5(weights=self.weights)
        self.preprocess = self.weights.transforms()

        # Fully connected layers
        self.fc1 = nn.Linear(1000, 512) # For Pre-trained Weights
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 14)  # 14-dimensional output

        self.preprocess_img = transforms.Compose([
            transforms.Resize((48, 64)),
            transforms.ToTensor()   
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)        

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class MobileNet_V3_Large(nn.Module):
    def __init__(self):
        super(MobileNet_V3_Large, self).__init__()

        # Try Mobile Net Large and Small
        self.weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.model = models.mobilenet_v3_large(weights=self.weights)
        self.preprocess = self.weights.transforms()

        # Fully connected layers
        self.fc1 = nn.Linear(1000, 512) # For Pre-trained Weights
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 14)  # 14-dimensional output

        self.preprocess_img = transforms.Compose([
            transforms.Resize((48, 64)),
            transforms.ToTensor()   
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)        

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class MobileNet_V3_Small(nn.Module):
    def __init__(self):
        super(MobileNet_V3_Small, self).__init__()

        # Try Mobile Net Large and Small
        self.weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.preprocess = self.weights.transforms()

        # Fully connected layers
        self.fc1 = nn.Linear(1000, 512) # For Pre-trained Weights
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 14)  # 14-dimensional output

        self.preprocess_img = transforms.Compose([
            transforms.Resize((48, 64)),
            transforms.ToTensor()   
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)        

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class LightDepthEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(LightDepthEncoder, self).__init__()
        
        self.cfg = cfg
        
        # the image goes through an initial convolution layer, before entering subsequent conv layers
        init_encoder_in_channels = cfg.initial_encoder_in_channels
        init_encoder_out_channels = cfg.initial_encoder_out_channels
        init_encoder_kernel_size = cfg.initial_encoder_kernel_size
        init_encoder_stride = cfg.initial_encoder_stride
        init_encoder_padding = cfg.initial_encoder_padding
        self.initial_conv = nn.Conv2d(init_encoder_in_channels, 
                                         init_encoder_out_channels, 
                                         init_encoder_kernel_size, 
                                         init_encoder_stride, 
                                         init_encoder_padding, 
                                         bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # creating the rest of the encoder, which is several convs + leaky relu layers
        self.convs = nn.ModuleList()
        num_enc_channels_in = cfg.num_enc_channels_in
        padding_required_for_same_size = self.get_padding_for_conv_same_output(cfg.kernel_size, cfg.stride)
                
        for i, (channel_in, channel_out) in enumerate(zip(num_enc_channels_in[:-1], num_enc_channels_in[1:])):
            layer = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding, bias=True),
                nn.LeakyReLU(cfg.leaky_relu_alpha, inplace=True),
                nn.Conv2d(channel_out, channel_out, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=padding_required_for_same_size, bias=True),
                nn.LeakyReLU(cfg.leaky_relu_alpha, inplace=True)
            )
            self.convs.append(layer)
        
        # initializing all conv layers using kaiming initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x) -> List[torch.Tensor]:
        # Visualization of forward pass. First pass through the initial convolution. 
        # Then, pass through all subsequent convolutional layers in the encoder.
        # |__|
        # |__|  --->  [INITIAL CONV -> RELU]  ---> IMAGE' ---> ENCODER (N * [CONV -> LEAKY RELU])
        # |__|
        # IMAGE                               
        # Returns: all intermediate features. 
        # Why return all features? Because decoder uses intermediate features in fusion.
        features = []
        x = self.initial_conv(x)
        x = self.relu(x)
        features.append(x)
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        return features
            
    def get_padding_for_conv_same_output(self, kernel_size, stride=1) -> int:
        # Given some kernel size and stride, calculate the padding to ensure that the 
        # output of the convolution is the same size as the input
        return ((stride - 1) + kernel_size - 1) // 2
    
class UpConvBlock(nn.Module):
    # Class Reference: https://github.com/Ecalpal/RT-MonoDepth/blob/main/networks/RTMonoDepth/RTMonoDepth.py
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.nonlin(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, num_in_ch: int):
        super(DecoderBlock, self).__init__()
        self.in_ch_sizes = [num_in_ch, 64, 32, 1]
        self.convs = []
        
        # creating the convolution layers for the decoder. has sizes [num_in_channels -> 64 -> 32 -> 1]
        for i in range(len(self.in_ch_sizes)-1):
            self.convs.append(nn.Conv2d(self.in_ch_sizes[i], self.in_ch_sizes[i+1], 3, 1, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, x):
        for i in range(len(self.convs)):
            if i == len(self.convs) - 1:
                continue
            x = self.leaky_relu(self.convs[i](x))
        return self.convs[-1](x)
            
class LightDepthDecoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(LightDepthDecoder, self).__init__()
        
        self.cfg = cfg
        
        # this is the sizes of the channels for the encoder (e.g., [64, 64, 128, 256])
        self.num_ch_enc = cfg.num_enc_channels_in 
        self.scales = len(self.num_ch_enc)
        
        # checks for whether or not we have skip connections for fusion blocks
        self.use_skips = cfg.use_skips
        
        # storing the decoder conv sizes (e.g., [16, 32, 64, 128, 256])
        self.num_dec_channels_in = cfg.num_dec_channels_in
        
        self.convs = OrderedDict()
        
        for i in range(len(self.num_dec_channels_in)-1, -1, -1):
            # each UpConv block has two layers. 
            # Layer one will reduce the dimension of the input (e.g., 128 -> 64)
            # Layer two is there to learn more parameters (e.g., 64 -> 64)
            
            # creating "layer one" of the upconv block
            num_channels_in = self.num_dec_channels_in[i+1]
            num_channels_out = self.num_dec_channels_in[i]
            self.convs[("upconv", i, 0)] = UpConvBlock(num_channels_in, num_channels_out)
            
            # creating "layer two" of the upconv block 
            num_channels_in = self.num_dec_channels_in[i]
            # condition: checks for if we apply a concatenation fusion instead of element-wise addition for the fusion
            if self.use_skips and i==1:
                num_channels_in += self.num_ch_enc[i-1]
            num_channels_out = self.num_dec_channels_in[i]
            self.convs[("upconv", i, 1)] = UpConvBlock(num_channels_in, num_channels_out)
        
        for i in self.scales:
            self.convs[("dispconv", i)] = DecoderBlock(self.num_dec_channels_in[i])
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features: List[torch.Tensor]) -> Dict[Tuple[str, int], torch.Tensor]:
        self.outputs = {}
        
        # recall: input features stores all intermediate features from the encoder
        # input_features[-1] is the last feature from the encoder
        x = input_features[-1]
        for i in range(len(self.num_dec_channels_in)-1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            
            # performing interpolation to increase the resolution by a scale factor of 2
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            
            if self.use_skips and i > 1:
                # fusion: addition
                x += input_features[i-1]
            elif self.use_skips and i == 1:
                # fusion: concatenation
                x = torch.cat([x, input_features[i-1]], 1)
            
            x = self.convs[("upconv", i, 1)](x)
            for i in self.scales:
                depth = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = depth
        
        return self.outputs                
    
class LightDepthModel(nn.Module):
    # Ref Implementation: https://github.com/Ecalpal/RT-MonoDepth/blob/main/networks/RTMonoDepth/RTMonoDepth.py
    def __init__(self, cfg: DictConfig) -> None:
        super(LightDepthModel, self).__init__()
        self.encoder = LightDepthEncoder(cfg)
        self.decoder = LightDepthDecoder(cfg)
    
    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x[('disp', 0)]