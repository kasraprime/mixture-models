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
    X = []
    X.append(0)
    for i in range(K-1):
        X.append(np.random.uniform(0,1))
    X.append(1)
    X.sort()
    Z = []
    for i in range(K):
        Z.append(X[i+1] - X[i])

    return Z



def ComputePosterior(data_i, component_k, pi, theta, K_mixture):
        theta[component_k]
        numerator = 0 
        denominator = 0        
        current_posterior_gamma_ik = 0

        # Computing numerator
        for d in range(len(data_i)):
            numerator = numerator * ( (theta[component_k][d]**data_i[d]) * ( (1 - theta[component_k][d])**(1 - data_i[d]) ) )
        
        # Computing denominator
        for k in range(K_mixture):
            temp = 0
            for d in range(len(data_i)):
                temp = temp * ( (theta[k][d]**data_i[d]) * ( (1 - theta[k][d])**(1 - data_i[d]) ) )
            denominator = denominator + (pi[k] * temp)

        current_posterior_gamma_ik = (pi[component_k] * numerator) / denominator
        return current_posterior_gamma_ik



def ComputeMarginal(K_mixture, train_loader, pi, theta):
    marginal = 0
    for i,data in enumerate(train_loader):
        data_i = torch.squeeze(data[0])
        data_i = data_i.view(-1)
        sum_k = 0
        for k in range(K_mixture):
            temp = 0
            for d in range(len(data_i)):
                temp = temp * ( (theta[k][d]**data_i[d]) * ( (1 - theta[k][d])**(1 - data_i[d]) ) )
            sum_k = sum_k + (pi[k] * temp)
        marginal = marginal * sum_k
    
    return marginal






def train(data_type, epoch_num, batch_size, K_mixture, J_parameter_dimension):
    """pi is a vector of length K, theta is of shape K*J, and in my case K is 10 and J=D=784"""
    train_loader, test_loader = mnist_loader(data_type, batch_size)

    # initializing pi in simplex(K-1), and theta of shape K*J
    pi = Simplex(K_mixture)
    theta = np.random.uniform(0,1,(K_mixture, J_parameter_dimension))
    alpha = np.random.uniform(0,1,(K_mixture))
    marginal_log_like = []
    epoch_list = []
    
    for epoch in tqdm(range(epoch_num)):
        theta_numerator = np.zeros((K_mixture, J_parameter_dimension))
        theta_denominator = np.zeros((K_mixture, J_parameter_dimension))

        pi_numerator = np.zeros((K_mixture))        

        for i,data in enumerate(train_loader):
            # data[0] is the batch_size*1*28*28 matrix and data[1] is the label
            # removing dimensions of size 1
            data_i = torch.squeeze(data[0])
            # convert the shape of tensor from 28*28 to 784
            data_i = data_i.view(-1)
            # data_i[d] represents x_{i,d}
            for k in range(K_mixture):
                # we pass theta which is parameters to compute current posterior
                posterior_gamma_ik = ComputePosterior(data_i, k, pi, theta, K_mixture)
                pi_numerator[k] = pi_numerator[k] + posterior_gamma_ik
                for d in range(J_parameter_dimension):
                    # is it okay if I use posterior_gamma_ik that is computed outside this for?
                    #posterior_gamma_ik = ComputePosterior(data_i, k, pi, theta)
                    theta_numerator[k][d] = theta_numerator[k][d] + (posterior_gamma_ik * data_i[d])
                    theta_denominator[k][d] = theta_denominator[k][d] + posterior_gamma_ik

        # Now that we have gone through all the data, we can update parameters:
        theta = theta_numerator / theta_denominator # Shall I have a loop or it works in python
        pi = (pi_numerator + alpha) / ( sum(alpha) + len(train_loader)) # Shall I have a loop or it works in python

        epoch_list.append(epoch)
        marginal = ComputeMarginal(K_mixture, train_loader, pi, theta)
        marginal_log_like.append(marginal)

    # plot marginal log likelihood for each epoch
    plt.plot(epoch_list,marginal_log_like)
    plt.title('Marginal Log Likelihood')
    plt.xlabel('epoch')
    plt.ylabel('marginal log likelihood')
    plt.savefig('Marginal.pdf')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='binary', type=str)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--K_mixture', default=10, type=int)
    parser.add_argument('--J_parameter_dimension', default=784, type=int)
    args = parser.parse_args()
    train(args.data_type,args.epoch_num,args.batch_size,args.K_mixture,args.J_parameter_dimension)