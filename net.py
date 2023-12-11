import torch.nn as nn
from Jtools import *

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class MLPNet(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super(MLPNet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        out = self.dense(x)
        return out
    
class LRNet(nn.Module):
    def __init__(self, input_size=3, output_size=1):
        super(LRNet, self).__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dense(x)
        out = self.sigmoid(out)
        return out
    
class UNet(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super(UNet, self).__init__()
        self.dconv_down0 = double_conv(input_size, 32)
        self.dconv_down1 = double_conv(32, 64)
        self.dconv_down2 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up1 = double_conv(128, 64)
        self.dconv_up0 = double_conv(64, 32)

        self.upsample1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.upsample0 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

        self.conv_last = nn.Conv2d(32, output_size, 1)

    def forward(self, x):
        conv0 = self.dconv_down0(x) 
        x = self.maxpool(conv0) 

        conv1 = self.dconv_down1(x) 
        x = self.maxpool(conv1) 
        x = self.dconv_down2(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        x = self.upsample0(x)
        x = torch.cat([x, conv0], dim=1)

        x = self.dconv_up0(x)
        out = self.conv_last(x)

        return out