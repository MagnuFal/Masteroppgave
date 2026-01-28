import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.doubleconv(x)

class DownPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
    
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.downpass = nn.Sequential(
            DoubleConvolution(in_channels=in_channels, out_channels=out_channels),
            self.maxpool()
        )

    def forward(self, x):
        self.downpass(x)

class UpPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels / 2, kernel_size=2, stride=2)
        self.double = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        self.up

class UNet(nn.Module):
    def __init__(self, n_input_channels = 1, n_classes = 3):
        super(UNet, self).__init__()

        self.double_conv1 = DoubleConvolution(n_input_channels, 64)
        self.down_pass1 = DownPass(64, 128)
        self.down_pass2 = DownPass(128, 256)
        self.down_pass3 = DownPass(256, 512)
        self.down_pass4 = DownPass(512, 1024)