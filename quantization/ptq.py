import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy

import os 
from pathlib import Path

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root = './', train = True, transform = transform, download = True)
test_set = torchvision.datasets.MNIST(root = './', train = False, transform = transform, download = True)
trainloader = DataLoader(train_set, batch_size = 8, shuffle = True)
testloader = DataLoader(test_set, batch_size = 8, shuffle = False)

