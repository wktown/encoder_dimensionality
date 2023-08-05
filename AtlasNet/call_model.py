p = '/home/wtownle1/dimensionality_powerlaw/activation_models/AtlasNet/'
import sys
sys.path.append(p)
from AtlasNet.model_2L import EngineeredModel2L

model = EngineeredModel2L(filters_2=1000).Build()
identifier = 'AtlasNet'
from activation_models.generators import wrap_pt
model = wrap_pt(model,identifier)

model.get_layer('c2')