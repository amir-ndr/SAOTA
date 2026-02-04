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
# model_cifar.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out, inplace=False)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = out + identity
#         out = F.relu(out, inplace=False)
#         return out


# class ResNetCIFAR(nn.Module):
#     """
#     ResNet for CIFAR-sized images (32x32):
#       - 3x3 conv stem (stride=1), no maxpool
#       - layers: [2,2,2,2] => ResNet-18
#       - global average pool -> fc
#     """

#     def __init__(self, block, layers, num_classes=10):
#         super().__init__()
#         self.in_planes = 64

#         # CIFAR stem: 3x3, stride 1, no maxpool
#         self.conv1 = nn.Conv2d(
#             3, 64, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(64)

#         self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         self._init_weights()

#     def _make_layer(self, block, planes, num_blocks, stride):
#         downsample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_planes,
#                     planes * block.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.in_planes, planes, stride=stride, downsample=downsample))
#         self.in_planes = planes * block.expansion
#         for _ in range(1, num_blocks):
#             layers.append(block(self.in_planes, planes, stride=1, downsample=None))

#         return nn.Sequential(*layers)

#     def _init_weights(self):
#         # Kaiming init for conv, default for BN/Linear is fine
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0.0, 0.01)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x, inplace=False)

#         x = self.layer1(x)  # 32x32
#         x = self.layer2(x)  # 16x16
#         x = self.layer3(x)  # 8x8
#         x = self.layer4(x)  # 4x4

#         x = self.avgpool(x)  # 1x1
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


# def resnet18_cifar(num_classes=10):
#     """Factory for CIFAR-style ResNet-18."""
#     return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# # Optional: keep your older name for compatibility with existing imports
# # If your code expects CNNCifar10(), you can alias it like this:
# class CNNCifar10(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.net = resnet18_cifar(num_classes=num_classes)

#     def forward(self, x):
#         return self.net(x)
