import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace
import random as rand

from torch import linalg
seed = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)

def loss_fn(state_embed, concept_embed, y):
    batch_size = state_embed.size(0)
    difference = state_embed - concept_embed 
    l2_norm = torch.linalg.norm(difference, dim=1, ord=2)
    #l2_norm = torch.sum(((state_embed * concept_embed)) ** 2, 1).sqrt()
    loss  = torch.sum((l2_norm-y) **2,0)/batch_size
        
    return loss
    
    
    
    
    