import torch.nn as nn
import torch.nn.functional as F
from itertools import pairwise
from omegaconf import DictConfig

class MLP(nn.Module):

    def __init__(self, cfg:DictConfig):

        # units, act, drop=None

        super(MLP, self).__init__()

        self.cfg = cfg
        self.linear_layers = nn.ModuleList([nn.Linear(a, b) for a, b in pairwise(cfg.units)])
        self.act = getattr(F, cfg.act)
        self.out_act = getattr(F, cfg.out_act) if cfg.out_act else None
        self.drop = nn.Dropout(cfg.drop) if cfg.drop else None

    def forward(self, x):
        
        for linear in self.linear_layers[:-1]:
            
            x = linear(x)
            x = self.act(x)
            if self.drop is not None:
                x = self.drop(x)
        
        x = self.linear_layers[-1](x)
        if self.out_act is not None:
            x = self.out_act(x)        
        
        return x