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
import os
from tqdm import tqdm

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x2 += self.shortcut(x)
        x2 = F.relu(x2)
        return x2

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dequant(out)
        return out

def train(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss/(i+1), accuracy=100.*correct/total)
        print(f'Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.3f}, Accuracy = {100.*correct/total:.2f}%')

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    inference_time = 0
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluation")
    with torch.no_grad():
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(inputs)
            inference_time += time.time() - start_time
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(accuracy=100.*correct/total)
    accuracy = 100. * correct / total
    avg_inference_time = inference_time / len(test_loader)
    return accuracy, avg_inference_time

def get_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    model = ResNet(Block, [2, 2, 2, 2])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    print("Training standard model...")
    train(model, train_loader, criterion, optimizer, device)
    standard_accuracy, standard_inference_time = evaluate(model, test_loader, device)
    standard_size = get_model_size(model)

    print("\nPreparing QAT model...")
    qat_model = copy.deepcopy(model)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    qat_model = prepare_qat(qat_model)
    qat_model = qat_model.to(device)

    print("Training QAT model...")
    optimizer = optim.SGD(qat_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    train(qat_model, train_loader, criterion, optimizer, device, epochs=5)

    qat_model = qat_model.cpu()
    qat_model = convert(qat_model)
    qat_accuracy, qat_inference_time = evaluate(qat_model, test_loader, torch.device('cpu'))
    qat_size = get_model_size(qat_model)

    print("\nComparison Results:")
    print(f"Standard Model - Accuracy: {standard_accuracy:.2f}%, Inference Time: {standard_inference_time*1000:.2f}ms, Size: {standard_size:.2f}MB")
    print(f"QAT Model     - Accuracy: {qat_accuracy:.2f}%, Inference Time: {qat_inference_time*1000:.2f}ms, Size: {qat_size:.2f}MB")


if __name__ == "__main__":
    main()