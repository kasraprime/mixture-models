import argparse
import math
import os
import pickle
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

epsilon = 1e-7

def mnist_loader(data_type,batch_size):
    """Setup MNIST data loader."""
    
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
    
    return train_loader, test_loader, len(train_dataset)


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


def ComputePosterior(data_i, component_k, pi, theta, K_mixture, J_parameter_dimension, device):
    current_posterior_gamma_ik = 0.0

    # Computing numerator        
    numerator =  np.prod((theta[component_k]**data_i)) * np.prod(( (1 - theta[component_k])**(1 - data_i) ) )
    numerator = pi[component_k] * numerator

    # Computing denominator
    denominator = 0.0
    for k in range(K_mixture):
        temp = np.prod((theta[k]**data_i)) * np.prod(( (1 - theta[k])**(1 - data_i) ))
        denominator = denominator + (pi[k] * temp)

    current_posterior_gamma_ik = numerator / (denominator + epsilon)
    return current_posterior_gamma_ik


def ComputeLogPosterior(data_i, component_k, pi, theta, K_mixture, J_parameter_dimension, device):

    current_posterior_gamma_ik = 0.0

    # Computing numerator
    numerator = 1.0
    numerator = np.add (np.matmul(data_i,np.log(theta[component_k])) , np.matmul((1-data_i),(np.log(1-theta[component_k]))))
    numerator = np.log(pi[component_k]) + numerator

    # Computing denominator
    denominator = 0.0
    for k in range(K_mixture):
        temp = np.add (np.matmul(data_i,np.log(theta[k])) , np.matmul((1-data_i),(np.log(1-theta[k]))))
        denominator = denominator + (pi[k] * temp)
    
    current_posterior_gamma_ik = numerator - np.log(denominator)
    return current_posterior_gamma_ik,numerator,denominator


def ComputeMarginal(K_mixture, J_parameter_dimension, train_loader, pi, theta, device, batch_size):
    marginal = 0.0
    for i,data in enumerate(train_loader):
        #data_i = data[0].to(device)
        data_i = data[0]
        data_i = torch.squeeze(data_i)
        data_i = data_i.view(-1)
        data_i = data_i.numpy()
        sum_k = 0.0
        for k in range(K_mixture):
            temp = np.prod((theta[k]**data_i)) * np.prod(( (1 - theta[k])**(1 - data_i) ))
            sum_k = sum_k + (pi[k] * temp)
        marginal = marginal + np.log(sum_k)
    
    return marginal


def train(data_type, epoch_num, batch_size, K_mixture, J_parameter_dimension, device, results):
    """pi is a vector of length K, theta is of shape K*J, and in my case K is 10 and J=D=784"""
    train_loader, test_loader, N_number_data = mnist_loader(data_type, batch_size)

    # initializing pi in simplex(K-1), and theta of shape K*J
    pi = Simplex(K_mixture)
    theta = np.random.uniform(0,1,(K_mixture, J_parameter_dimension))
    #alpha = np.random.uniform(0,1,(K_mixture))
    alpha_constant = np.random.uniform(0,1)
    alpha = np.full(K_mixture, alpha_constant)
    marginal_log_like = []
    epoch_list = []
    
    for epoch in tqdm(range(epoch_num)):
        theta_numerator = np.zeros((K_mixture, J_parameter_dimension))
        theta_denominator = np.zeros((K_mixture, J_parameter_dimension))

        pi_numerator = np.zeros((K_mixture))        

        for i,data in enumerate(train_loader):
            # data[0] is the batch_size*1*28*28 matrix and data[1] is the label
            # removing dimensions of size 1
            #data_i = data[0].to(device)
            data_i = data[0]
            data_i = torch.squeeze(data_i)
            # convert the shape of tensor from 28*28 to 784
            data_i = data_i.view(-1)
            data_i = data_i.numpy()
            # data_i[d] represents x_{i,d}
            for k in range(K_mixture):
                # we pass theta which is parameters to compute current posterior
                posterior_gamma_ik = ComputeLogPosterior(data_i, k, pi, theta, K_mixture, J_parameter_dimension, device)
                pi_numerator[k] = pi_numerator[k] + posterior_gamma_ik
                theta_numerator[k] = theta_numerator[k] + (posterior_gamma_ik * data_i)
                theta_denominator[k] = theta_denominator[k] + posterior_gamma_ik
            if ((i+1)*batch_size)%2000 == 0:
                print('epoch:', epoch+1, 'data processed so far:', (i+1)*batch_size)

        # Now that we have gone through all the data, we can update parameters:
        theta = theta_numerator / (theta_denominator + epsilon)
        pi = (pi_numerator + alpha - 1) / ( sum(alpha) + N_number_data - K_mixture + epsilon)

        epoch_list.append(epoch+1)
        marginal = ComputeMarginal(K_mixture, J_parameter_dimension, train_loader, pi, theta, device, batch_size)
        print('epoch:', epoch+1, 'marginal log likelihood:', marginal )
        marginal_log_like.append(marginal)
        pickle.dump( ([epoch_list, marginal_log_like]), open( results+'epoch_marginal.pkl', "wb" ) )

    # plot marginal log likelihood for each epoch
    plt.plot(epoch_list,marginal_log_like)
    plt.title('Marginal Log Likelihood')
    plt.xlabel('epoch')
    plt.ylabel('marginal log likelihood')
    plt.savefig(results+'Marginal.pdf')

    #saving the marginal after training is done to compare to other experiments
    pickle.dump( (marginal_log_like[-1]), open( results+'finalmarginal.pkl', "wb" ) )

    #saving the parameters of the model to able to replicate the results
    pickle.dump( (pi), open( results+'pi.pkl', "wb" ) )
    pickle.dump( (theta), open( results+'theta.pkl', "wb" ) )
    pickle.dump( (alpha), open( results+'alpha.pkl', "wb" ) )

    # Note that we have K components each of which are J dimensional where J=D and D is the number of pixels. So we can show images (instances) based on each K component
    # saving some instances per component.
    all_sampled_data = []    
    for k in range(K_mixture):
        for instance in range(3):
            sampled_data = np.zeros(J_parameter_dimension)
            for d in range(J_parameter_dimension):
                sampled_data[d] = np.random.binomial(size=1, n=1, p= theta[k][d])
            sampled_data = sampled_data.reshape(28,28)
            matplotlib.image.imsave(results+'component'+str(k)+'instance'+str(instance)+'.png', sampled_data) 
            all_sampled_data.append(sampled_data)

    pickle.dump( (all_sampled_data), open( results+'sampledImages.pkl', "wb" ) )        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='Random', type=str)
    parser.add_argument('--data_type', default='binary', type=str)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--K_mixture', default=10, type=int)
    parser.add_argument('--J_parameter_dimension', default=784, type=int)
    args = parser.parse_args()
    device_name = 'cuda:'+str(0) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    results = './results/' + args.experiment_name + '/'
    if not os.path.exists(results):
        os.makedirs(results)

    print('device in use:', device)
    train(args.data_type,args.epoch_num,args.batch_size,args.K_mixture,args.J_parameter_dimension, device, results)