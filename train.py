import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm

def mnist_loader():    
    """Setup MNIST data loaders."""
    
    # Load datasets.
    train_dataset = MNIST('data/pytorch_data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_dataset = MNIST('data/pytorch_data/MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    
    kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': 32, 'shuffle': False} 
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    
    return train_loader, test_loader

train_loader, test_loader = mnist_loader()

for epoch in tqdm(range(10)):
    for data in train_loader:
