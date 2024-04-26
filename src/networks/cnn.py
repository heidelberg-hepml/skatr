import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint

class CNN(nn.Module):
    
    def __init__(self, cfg:DictConfig):
        super().__init__()

        self.cfg = cfg
        in_ch, hi_ch, ou_ch = cfg.in_channels, cfg.hidden_channels, cfg.out_channels

        self.block1 = ConvBlock(
            nn.Conv3d(in_ch, hi_ch, kernel_size=(3,3,cfg.zstride), stride=(1,1,cfg.zstride)),
            nn.Conv3d(hi_ch, hi_ch, kernel_size=(3,3,2)),
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            checkpoint=cfg.checkpoint_grads
        )

        self.block2 = ConvBlock(
            nn.Conv3d(hi_ch, 2*hi_ch, kernel_size=(3,3,2)),
            nn.Conv3d(2*hi_ch, 2*hi_ch, kernel_size=(3,3,2), padding=(1,1,0)),
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1)),
            checkpoint=cfg.checkpoint_grads
        )

        self.block3 = ConvBlock(
            nn.Conv3d(2*hi_ch, 4*hi_ch, kernel_size=(3,3,2)),
            nn.Conv3d(4*hi_ch, 4*hi_ch, kernel_size=(3,3,2), padding=(1,1,0)),
            nn.AvgPool3d(kernel_size=tuple(cfg.pool_size)),
            checkpoint=cfg.checkpoint_grads
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,128)
        self.out = nn.Linear(128, ou_ch)
        
        self.out_act = getattr(F, cfg.out_act) if cfg.out_act else None
    
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
       
        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
       
        x = self.out(x)
        if self.cfg.out_act:
            x = self.out_act(x)
        return x
    
# TODO: reimplement gradient checkpointing
class ConvBlock(nn.Module):

    def __init__(self, conv1, conv2, pool, checkpoint=False):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.pool = pool
        self.checkpoint=checkpoint
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x)