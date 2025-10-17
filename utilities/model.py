import torch.nn as nn
import torch
import os
import json
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))

class FE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        x = self.fc3(x)
        return x
    
def simplecnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)


class FashionCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FashionCNN, self).__init__()
        self.base = FashionFE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))

class FashionFE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FashionFE, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
    
def fashioncnn(n_classes):
    return FashionCNN(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
        
class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        x = self.fc3(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, inchannels=3):
        super(resnet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(inchannels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(n_classes):
    return resnet(BasicBlock, [2, 2, 2, 2], n_classes)

def resnet18_fmnist(n_classes):
    return resnet(BasicBlock, [2, 2, 2, 2], n_classes, inchannels=1)

class ResNetBase(nn.Module):
    def __init__(self, block, num_blocks, inchannels=3):
        super(ResNetBase, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(inchannels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, inchannels=3):
        super(ResNet, self).__init__()
        self.base = ResNetBase(block, num_blocks, inchannels)
        self.classifier = ResNetClassifier(512 * block.expansion, num_classes)  

    def forward(self, x):
        features = self.base(x)
        logits = self.classifier(features)
        return logits
def ResNet18(n_classes, inchannels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, inchannels)

def ResNet18_fmnist(n_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, inchannels=1)

