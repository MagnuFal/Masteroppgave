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
        return self.doubleconv(x)

class DownPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
    
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.downpass = nn.Sequential(
            self.maxpool,
            DoubleConvolution(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.downpass(x)

class UpPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]

        x1_pad = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

        x_cat = torch.cat([x2, x1_pad], dim = 1)

        return self.conv(x_cat)
    
class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_input_channels = 1, n_classes = 3):
        super(UNet, self).__init__()

        self.double_conv1 = DoubleConvolution(n_input_channels, 64)
        self.down_pass1 = DownPass(64, 128)
        self.down_pass2 = DownPass(128, 256)
        self.down_pass3 = DownPass(256, 512)
        self.down_pass4 = DownPass(512, 1024)
        self.up_pass1 = UpPass(1024, 512)
        self.up_pass2 = UpPass(512, 256)
        self.up_pass3 = UpPass(256, 128)
        self.up_pass4 = UpPass(128, 64)
        self.double_conv2 = FinalConv(64, n_classes)

    def forward(self, x):
        x1 = self.double_conv1(x)
        x2 = self.down_pass1(x1)
        x3 = self.down_pass2(x2)
        x4 = self.down_pass3(x3)
        x5 = self.down_pass4(x4)
        x6 = self.up_pass1(x5, x4)
        x7 = self.up_pass2(x6, x3)
        x8 = self.up_pass3(x7, x2)
        x9 = self.up_pass4(x8, x1)
        x10 = self.double_conv2(x9)

        return x10