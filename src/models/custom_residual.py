import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0):
        super().__init__()
        mid = out_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.drop(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class CustomResidualNet(nn.Module):
    def __init__(self, num_classes=10, width=64, drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            BottleneckBlock(width, width, stride=1, drop=drop),
            BottleneckBlock(width, width, stride=1, drop=drop),
        )
        self.stage2 = nn.Sequential(
            BottleneckBlock(width, width*2, stride=2, drop=drop),
            BottleneckBlock(width*2, width*2, stride=1, drop=drop),
        )
        self.stage3 = nn.Sequential(
            BottleneckBlock(width*2, width*4, stride=2, drop=drop),
            BottleneckBlock(width*4, width*4, stride=1, drop=drop),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width*4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
