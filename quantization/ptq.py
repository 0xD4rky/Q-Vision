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

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear((28*28),512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optim = Adam(model.parameters(),lr = 0.001)

from tqdm import tqdm

def train(model, trainloader,epoch, epochs, criterion, optimizer, device):

  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for i,batch in enumerate(tqdm(trainloader, desc = f"training {epoch+1}/{epochs} epochs")):

    x,y = batch
    x = x.to(device)
    y = y.to(device)

    gradients = []
    output = model(x)
    optimizer.zero_grad()
    loss = criterion(output,y)
    loss.backward()

    for param in model.parameters():
      gradients.append(param.grad.detach().cpu().numpy().copy())

    optimizer.step()

    running_loss += loss.item()
    _, prediction = output.max(1)
    total += y.size(0)
    correct += prediction.eq(y).sum().item() #.item() is used to extract scalar value from tensor

  train_loss = running_loss / len(trainloader)
  train_accuracy = 100 * (correct/total)
  return train_loss, train_accuracy, gradients

def eval(model, testloader, epoch, epochs, criterion, device):

  model.eval()
  running_loss = 0.0
  total = 0
  correct = 0

  with torch.no_grad():

    for i, batch in enumerate(tqdm(testloader)):

      x,y = batch
      x = x.to(device)
      y = y.to(device)

      output = model(x)
      loss = criterion(output,y)

      running_loss += loss.item() # extracts scalar from tensor
      _, prediction = output.max(1)
      total += y.size(0)
      correct += prediction.eq(y).sum().item()

    
    test_loss = running_loss / len(testloader)
    test_accuracy = 100 * (correct/total)
    return test_loss, test_accuracy
  
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import time

for epoch in range(epochs):
  start = time.time()
  loss, acc, gradients = train(model, trainloader, epoch, epochs, criterion, optim, device)
  end = time.time()
  print(f"time taken to run {epoch+1}th epoch is : {(end-start)}")
  print(f"training loss : {loss}")
  print(f"training accuracy : {acc}")
  print(F"Training complete")

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import time

for epoch in range(epochs):
  start = time.time()
  loss, acc = eval(model, testloader, epoch, epochs, criterion, device)
  end = time.time()
  print(f"time taken to run {epoch+1}th epoch is : {(end-start)}")
  print(f"testing loss : {loss}")
  print(f"testing accuracy : {acc}")
  print(F"testing complete")

import os
from pathlib import Path
def print_size_of_model(model):

  torch.save(model.state_dict(),"unquantized.p")
  print("size (KB) ", os.path.getsize("unquantized.p")/1e3)
  os.remove("unquantized.p")

model_name = "unquantized.pt"

if Path(model_name).exists():
  model.load_state_dict(torch.load(model_name))
  print("model loaded from disk")
else:
  torch.save(model.state_dict(), model_name)


def before_quant():
    """
    fund: to analyse the weights and model sizes before quantization
    param: no param required
    """
    print('weights before quantization')
    print(model[1].weight)
    print(model[1].weight.dtype)
    print('size of model before quantization :')
    print_size_of_model(model)


class Quantized_model(nn.Module):
  """
  aim: this class is used to create a quantized model using a min max observer
  """
  def __init__(self):
    super(Quantized_model,self).__init__()
    self.quant = torch.quantization.QuantStub()
    self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear((28 * 28), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    self.dequant = torch.quantization.DeQuantStub()

  def forward(self,x):
    x = self.quant(x)
    x = self.model(x)
    x = self.dequant(x)
    return x

def remap_state_dict(original_state_dict, quantized_model):
    new_state_dict = {}
    for key in original_state_dict.keys():
        new_key = "model." + key if key.startswith(("1.", "3.", "5.", "7.")) else key
        new_state_dict[new_key] = original_state_dict[key]
    return new_state_dict

quantized_model = Quantized_model()
original_state_dict = model.state_dict()
mapped_state_dict = remap_state_dict(original_state_dict, quantized_model)
quantized_model.load_state_dict(mapped_state_dict)
quantized_model.to(device)
quantized_model.eval()

quantized_model.qconfig = torch.ao.quantization.default_qconfig
quantized_model = torch.ao.quantization.prepare(quantized_model)

epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import time

def test(epochs):
  for epoch in range(epochs):
    start = time.time()
    loss, acc = eval(quantized_model, testloader, epoch, epochs, criterion, device)
    end = time.time()
    print(f"time taken to run {epoch+1}th epoch is : {(end-start)}")
    print(f"testing loss : {loss}")
    print(f"testing accuracy : {acc}")
    print(F"testing complete")

test(epochs)

quantized_model = torch.ao.quantization.convert(quantized_model)
print(f"weights after quantization is : ")
for i in [1]: 
    print(f"\nQuantizedLinear layer {i} weights:")
    print(torch.int_repr(quantized_model.model[i].weight))

def comparision(model, quantized_model):
  """
  Observing:
  * quantized weights
  * normal-weights
  * dequantized weights
  """

  layer = quantized_model.model[1]
  print(f"\nLayer {1} quantized weight: \n")
  weight_tensor = layer.weight()  
  print(torch.int_repr(weight_tensor))

  layer = model[1]
  print(f"\nlayer {1} un-quantized weights: \n")
  print(layer.weight)

  layer = quantized_model.model[1]
  print(f"\nlayer {1} de-quantized weights: \n")
  print(torch.dequantize(layer.weight()))