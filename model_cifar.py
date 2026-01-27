# model_cifar.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 32 -> 16 -> 8 -> 4 after 3 pools if we pool 3 times
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = self.pool(x)  # 16x16

        x = F.relu(self.conv3(x), inplace=False)
        x = F.relu(self.conv4(x), inplace=False)
        x = self.pool(x)  # 8x8

        x = F.relu(self.conv5(x), inplace=False)
        x = F.relu(self.conv6(x), inplace=False)
        x = self.pool(x)  # 4x4

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=False)
        x = self.fc2(x)
        return x
