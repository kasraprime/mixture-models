import argparse
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
import matplotlib.pyplot as plt

def mnist_loader(data_type,batch_size):    
    """Setup MNIST data loaders."""
    
    # Load datasets.
    if data_type == 'float':
        train_dataset = MNIST('data/pytorch_data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = MNIST('data/pytorch_data/MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif data_type == 'binary':
        train_dataset = MNIST('data/pytorch_data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda x:x>0, lambda x: x.float()]))
        test_dataset = MNIST('data/pytorch_data/MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda x:x>0, lambda x: x.float()]))

    kwargs = {'num_workers': 8, 'pin_memory': True, 'batch_size': batch_size, 'shuffle': False} 
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)
    
    return train_loader, test_loader

def train(data_type,epoch_num,batch_size):

    train_loader, test_loader = mnist_loader(data_type,batch_size)

    







if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='float', type=str)
    parser.add_argument('--epoch_num', default=10, type=int)    
    parser.add_argument('--batch_size', default=1, type=int)    
    args = parser.parse_args()
    train(args.data_type,args.epoch_num,args.batch_size)