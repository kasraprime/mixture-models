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


def Simplex(K):    
    X=[]
    X.append(0)
    for i in range(K-1):
        X.append(np.random.uniform(0,1))
    X.append(1)
    X.sort()
    Z=[]
    for i in range(K):
        Z.append(X[i+1] - X[i])

    return Z



def ComputePosterior(data_i,component_k):    
        current_posterior=
        for d in range(len(data_i)):
            # data_i[d] represents x_{i,d}
            for k in range(K_mixture):




def train(data_type,epoch_num,batch_size,K_mixture):
    """pi is a vector of length K, theta is of shape K*J, and in my case K is 10 and J=D=784"""
    train_loader, test_loader = mnist_loader(data_type,batch_size)

    # initializing pi in simplex(K-1)
    pi=Simplex(K_mixture)
    
    for epoch in tqdm(range(epoch_num)):
        for i, data in enumerate(train_loader):
            # data[0] is the batch_size*1*28*28 matrix and data[1] is the label
            # removing dimensions of size 1
            data_temp=torch.squeeze(data[0])
            # convert the shape of tensor from 28*28 to 784
            data_flat=data_temp.view(-1)         
            current_posterior=ComputePosterior(data_flat,component_k,pi)
            current_posterior*
            #update theta_{k,d}
            theta_kd=


        #update pi_{k}

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='float', type=str)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--K_mixture', default=10, type=int)
    args = parser.parse_args()
    train(args.data_type,args.epoch_num,args.batch_size,args.K_mixture)