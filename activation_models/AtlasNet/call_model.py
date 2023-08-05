import sys

#p1 = '/home/wtownle1/encoder_dimensionality/Engineered_Models'
#p2 = '/home/wtownle1/encoder_dimensionality/Engineered_Models/models'
#p3 = '/home/wtownle1/encoder_dimensionality/Engineered_Models/models/layer_operations'
#sys.path.append(p1)
#sys.path.append(p2)
#sys.path.append(p3)

#from Engineered_Models.models.model_2L import EngineeredModel2L
#model = EngineeredModel2L(filters_2=1000).Build()

#p1 = '/home/wtownle1/encoder_dimensionality/activation_models'
#sys.path.append(p1)


from activation_models.AtlasNet.model_2L import EngineeredModel2L
#from activation_models.AtlasNet.model_2L_eig import EngineeredModel2L
from activation_models.AtlasNet.model_3L import EngineeredModel3L
import torch

torch.manual_seed(seed=0)

model = EngineeredModel2L(filters_2=1000, k_size=5).Build()

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #print(model.state_dict()[param_tensor])
    
#print(model.named_parameters())
#c2.weight

for name, param in model.c2.named_parameters():
    if name in ['weight']:
        print(param)