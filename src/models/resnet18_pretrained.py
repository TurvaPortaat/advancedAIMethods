import torch.nn as nn
import torchvision.models as models

class ResNet18Pretrained(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.resnet18(weights=weights)

        # CIFAR-friendly stem
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

        # Replace classifier
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)
