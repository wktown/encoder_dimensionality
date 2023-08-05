#from engineered_model import Model
from layer_operations.convolution import StandardConvolution,RandomProjections
from layer_operations.output import Output

from layer_operations.convolution import *
from layer_operations.output import Output
import torch
from torch import nn



class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                mp2: nn.Module,
                c3: nn.Module,
                mp3: nn.Module,
                batches_3: int,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.mp1 = mp1
        self.c2 = c2
        self.mp2 = mp2
        self.c3 = c3
        self.mp3 = mp3
        self.batches_3 = batches_3
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        #conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
    
        x = self.mp1(x)
        if self.print_shape:
            print('mp1', x.shape)
            
        #conv layer 2
        x = self.c2(x)
        if self.print_shape:
            print('conv2', x.shape)        
        if self.print_shape:
            print('mp3', x.shape)
            
        x = self.mp2(x)
        if self.print_shape:
            print('mp2', x.shape)
            
        #conv layer 3
        conv_3 = []
        for i in range(self.batches_3):
            conv_3.append(self.c3(x)) 
        x = torch.cat(conv_3,dim=1)
        if self.print_shape:
            print('conv3', x.shape)
            
        x = self.mp3(x)
        if self.print_shape:
            print('mp3', x.shape)
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    



  
class EngineeredModel3L:
    
    
    def __init__(self, curv_params = {'n_ories':8,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=2000,filters_3=10000,batches_3 = 1):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=15,filter_type='curvature',curv_params=self.curv_params)     
        mp1 = nn.MaxPool2d(kernel_size=3)
        c2 = nn.Conv2d(24, self.filters_2, kernel_size=(9, 9))
        mp2 = nn.MaxPool2d(kernel_size=2)
        c3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7))
        mp3 = nn.MaxPool2d(kernel_size=2)

        last = Output()

        return Model(c1,mp1,c2,mp2,c3,mp3,self.batches_3,last)  
    