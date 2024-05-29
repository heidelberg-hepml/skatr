import os
import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig

from .. import networks

log = logging.getLogger(__name__)

class Model(nn.Module):

    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg

        # initialize network
        net_cls = getattr(networks, cfg.net.arch)
        self.net = net_cls(cfg.net)

        # # optionally initialize backbone
        # if cfg.backbone:
        #     bb_cls = getattr(networks, cfg.backbone.arch)
        #     self.bb = bb_cls(cfg.backbone)

    def load(self, exp_dir, device):
        path = os.path.join(exp_dir, 'model.pt')
        state_dicts = torch.load(path, map_location=device)
        self.load_state_dict(state_dicts["model"])
