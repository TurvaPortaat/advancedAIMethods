import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def get_dataloaders(batch_size=64, val_size=5000, seed=42, num_workers=2):

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4),transforms.ToTensor(),])

    test_transform = transforms.Compose([transforms.ToTensor()])


    trainset_all_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download = True, transform=train_transform)
    trainset_all_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)

    generator = torch.Generator().manual_seed(seed)
    train_size = len(trainset_all_aug)-val_size
    train_subset, val_subset_indices = random_split(range(len(trainset_all_aug)), [train_size, val_size], generator=generator)

    train_indices = list(train_subset)
    val_indices = list(val_subset_indices)

    trainset = torch.utils.data.Subset(trainset_all_aug, train_indices)
    valset = torch.utils.data.Subset(trainset_all_clean, val_indices)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return  trainset, trainloader, valset, valloader, testset, testloader


