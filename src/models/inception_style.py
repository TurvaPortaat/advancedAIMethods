import torch

class Inception_module(torch.nn.Module):
    def __init__(self):
        super(Inception_module, self).__init__()

        self.conv1x1 = torch.nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = torch.nn.Conv2d(192,128, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = torch.nn.Conv2d(192,32, kernel_size=5, stride=1, padding=2)

        self.maxpool = torch.nn.MaxPool2d(1, stride=1, padding=0)

        self.dim_red = torch.nn.Conv2d(192,32,kernel_size=1, stride=1,padding=0)

    def forward(self, x):

        x1 = torch.relu(self.conv1x1(x))
        x3 = torch.relu(self.conv3x3(x))
        x5 = torch.relu(self.conv5x5(x))

        xmaxp = self.maxpool(x)
        xmaxr = torch.relu(self.dim_red(xmaxp))

        x = torch.cat((x1, x3, x5, xmaxr), 1)

        return x

class InceptionCIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Muunna CIFAR (3 kanavaa) -> 192 kanavaa
        self.stem = torch.nn.Conv2d(3, 192, kernel_size=3, padding=1)

        # inception-palikka
        self.inception = Inception_module()

        # tee luokitus
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.stem(x)              # (B,3,32,32) -> (B,192,32,32)
        x = self.inception(x)         # -> (B,256,32,32)
        x = self.pool(x)              # -> (B,256,1,1)
        x = x.view(x.size(0), -1)     # -> (B,256)
        return self.fc(x)             # -> (B,10)


