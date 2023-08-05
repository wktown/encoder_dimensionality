from activation_models.AtlasNet.layer_operations.convolution import StandardConvolution,RandomProjections
from activation_models.AtlasNet.layer_operations.output import Output
from activation_models.AtlasNet.layer_operations.convolution import *
import torch
from torch import nn
import numpy as np
                       


class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                mp2: nn.Module,
                batches_2: int, 
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.mp1 = mp1
        self.c2 = c2
        self.mp2 = mp2
        self.batches_2 = batches_2
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        x = x.cuda()#conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
    
        x = self.mp1(x)
        if self.print_shape:
            print('mp1', x.shape)        
        
        #conv layer 2
        conv_2 = []
        for i in range(self.batches_2):
            conv_2.append(self.c2(x)) 
        x = torch.cat(conv_2,dim=1)
        if self.print_shape:
            print('conv2', x.shape)
            
        x = self.mp2(x)
        if self.print_shape:
            print('mp2', x.shape)            
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x 
    
    
class EngineeredModel2L:
    
    """
    Used to Initialize the Engineered Model
    
    Attributes
    ----------
    curv_params : dict
        the parameters used for creating the gabor filters. The number of filters in this layer = n_ories x n_curves x number of frequencies
    
    filters_2 : str
        number of random filters used in conv layer 2
    
    batches_2 : str 
        the number of batches used to apply conv layer 2 filters. Can be used for larger number of filters to avoid memory issues 
    """
    
    def __init__(self, curv_params = {'n_ories':8,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=5000,batches_2=1, k_size=9):
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.batches_2 = batches_2
        self.k_size = k_size
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=15,filter_type='curvature',curv_params=self.curv_params)     
        mp1 = nn.MaxPool2d(kernel_size=3)
        c2 = nn.Conv2d(24, self.filters_2, kernel_size=(self.k_size, self.k_size), device='cuda')
        with torch.no_grad():
            in_size = c2.weight.shape[1]
            c2_w = c2.weight.reshape(self.filters_2, self.k_size*self.k_size*in_size)
            
            np.random.seed(seed=0)

            # Set parameters
            num_filters = c2_w.shape[0]
            num_samples = c2_w.shape[-1]
            #num_samples = 5000  # Number of samples
            #num_features = 1000  # Number of features
            power_law_exponent = -1  # Exponent of power law decay
            variance_scale = 1  # Scaling factor for eigenvectors' variances

            # Generate principal components
            eigenvalues = np.power(np.arange(1, num_filters+1, dtype=float), power_law_exponent)
            eigenvectors = np.random.randn(num_filters, num_filters)
            eigenvectors = eigenvectors / np.sqrt(np.sum(np.square(eigenvectors), axis=0))  # Normalize columns

            # Scale eigenvectors' variances
            eigenvectors = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :] * variance_scale

            # Generate random data
            X = np.random.randn(num_samples, num_filters) @ eigenvectors
            c2_w = torch.tensor(np.transpose(X)).reshape(self.filters_2, in_size, self.k_size, self.k_size)
            c2.weight = torch.nn.Parameter(data= c2_w)
            
            # *bias is not the same eigenspectrum (?) *
            
        mp2 = nn.MaxPool2d(kernel_size=2)

        last = Output()

        return Model(c1,mp1,c2,mp2,self.batches_2,last)
    