import os
import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig

from src import networks
from src.utils.config import get_prev_config

log = logging.getLogger('Model')

class Model(nn.Module):

    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg

        # net_cls = getattr(networks, cfg.net.arch)
        # # TODO: Automatically set MLP input dim to backbone embedding dim
        # self.net = net_cls(cfg.net)

        # optionally initialize a backbone
        if cfg.backbone:
            log.info('Loading pretrained backbone')
            self.load_backbone()
            log.info(
                f'Backbone ([{self.bb.__class__.__name__}]) has '
                f'{sum(w.numel() for w in self.bb.parameters())} parameters'
            )

        if not cfg.replace_backbone:
            # initialize network
            net_cls = getattr(networks, cfg.net.arch)
            # TODO: Automatically set MLP input dim to backbone embedding dim
            self.net = net_cls(cfg.net)
        
    @property
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def update(self, optimizer, loss, step=None, total_steps=None):
        # propagate gradients
        loss.backward()
        # optionally clip gradients
        if clip := self.cfg.training.gradient_norm:
            nn.utils.clip_grad_norm_(self.trainable_parameters, clip)
        # update weights
        optimizer.step()

    def load(self, exp_dir, device):
        path = os.path.join(exp_dir, 'model.pt')
        state_dicts = torch.load(path, map_location=device)
        self.load_state_dict(state_dicts["model"])

    def load_backbone(self):
        
        bb_dir = self.cfg.backbone

        # load backbone state
        model_state = torch.load(os.path.join(bb_dir, 'model.pt'))["model"]
        net_state = {
            k.replace('net.', ''): v for k,v in model_state.items() if k.startswith('net.')
        }
        
        # read backbone config
        bcfg = get_prev_config(bb_dir)
        
        # initialize backbone net
        bb_cls = getattr(networks, bcfg.net.arch)
        self.bb = bb_cls(bcfg.net)
        self.bb.load_state_dict(net_state)
        
        if self.cfg.frozen_backbone:
            # freeze weights and set to eval mode
            for p in self.bb.parameters():
                p.requires_grad = False
            self.bb.eval()

        if self.cfg.adapt_res:
            self.bb.init_pos_grid(self.cfg.data_shape)

        if self.cfg.replace_backbone:
            
            # init new head or input adaption if needed
            if self.cfg.net.use_head: self.bb.init_head(self.cfg.net.head)
            if self.cfg.net.adapt_res: self.bb.init_adaptor(self.cfg.net.adaptor)
            
            self.net = self.bb