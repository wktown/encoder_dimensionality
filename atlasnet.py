p = '/home/wtownle1/dimensionality_powerlaw/activation_models/AtlasNet/'
import sys
sys.path.append(p)
from activation_models.AtlasNet.model_2L import EngineeredModel2L

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import alexnet

model = EngineeredModel2L(filters_2=1000).Build()
#model = alexnet(weights=None)
weights = []
n = 0
for param in model.parameters():
    n+=1
    if n > 1:
#    #if type(m).__name__ == 'c2':
        val=param.cpu()
        val = val.detach().numpy()
        val = np.ndarray.flatten(val)
        #val = np.ndarray.tolist(val)
        #weights.append(val)

#print(len(weights))
#w = pd.DataFrame(data=weights)
#w.to_csv('/home/wtownle1/encoder_dimensionality/tests/AtlasNetWeights')

min = np.amin(val)
max = np.amax(val)
mean = np.mean(val)

print(min)
print(max)
print(mean)