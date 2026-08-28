import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Fashion-MNIST statistics
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)

def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=FASHION_MNIST_MEAN, std=FASHION_MNIST_STD)
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=FASHION_MNIST_MEAN, std=FASHION_MNIST_STD)
    ])

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,  # Set to 0 on Windows local execution to prevent multiprocessing issues
) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )
    
    val_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader