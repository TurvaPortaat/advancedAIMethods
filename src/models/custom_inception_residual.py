import torch
import torch.nn as nn
import torch.nn.functional as F

class IncResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # split channels between branches
        b1 = out_ch // 4
        b3 = out_ch // 2
        b5 = out_ch // 8
        bp = out_ch - (b1 + b3 + b5)

        self.branch1 = nn.Conv2d(in_ch, b1, kernel_size=1, padding=0, bias=False)
        self.branch3 = nn.Conv2d(in_ch, b3, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, b5, kernel_size=5, padding=2, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_red = nn.Conv2d(in_ch, bp, kernel_size=1, padding=0, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        xp = self.pool_red(self.pool(x))

        out = torch.cat([x1, x3, x5, xp], dim=1)
        out = self.bn(out)

        out = out + self.skip(x)   # residual connection
        return F.relu(out)

class CustomInceptionResidualNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = IncResBlock(64, 128)
        self.block2 = IncResBlock(128, 256)

        self.down1 = nn.MaxPool2d(2)  # 32 -> 16
        self.block3 = IncResBlock(256, 256)
        self.down2 = nn.MaxPool2d(2)  # 16 -> 8
        self.block4 = IncResBlock(256, 512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.down1(x)
        x = self.block3(x)
        x = self.down2(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
