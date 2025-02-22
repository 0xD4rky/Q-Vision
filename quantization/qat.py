import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequantial()
        if stride != 1 or in_channels != self.expansion * in_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * in_channels, kernel_size = 1, stride = stride, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self,x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(x)
        x = F.relu(x)
        return x

