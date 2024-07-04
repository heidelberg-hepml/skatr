import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src import networks
from src.models.base_model import Model
from src.utils import augmentations, masks

class JEPA(Model):

    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.predictor = networks.PredictorViT(cfg.predictor)
        self.ctx_encoder = self.net
        self.tgt_encoder = self.net.__class__(cfg.net)
        self.augment = augmentations.RotateAndReflect()
        if cfg.norm_target:
            self.norm = nn.BatchNorm1d(cfg.hidden_dim)  # TODO: Remove norm?

        match cfg.sim:
            case 'l2':
                self.sim = lambda x1, x2: -F.mse_loss(x1, x2)
            case 'l1':
                self.sim = lambda x1, x2: -F.l1_loss(x1, x2)
            case 'smooth_l1':
                self.sim = lambda x1, x2: -F.smooth_l1_loss(x1, x2)

    def batch_loss(self, batch):        

        # augment batch
        x1 = batch[0]
        x2 = self.augment(x1) if self.cfg.augment else x1

        # sample masks
        num_patches = self.ctx_encoder.num_patches
        tgt_masks, ctx_masks = masks.JEPA_mask(
            num_patches, self.cfg.masking, batch_size=x2.size(0), device=x2.device
        )
            
        loss = 0.
        # WARNING: Assumes each target mask has it's own context. Repeat ctx_mask otherwise
        for ctx_mask, tgt_mask in zip(ctx_masks, tgt_masks):
            
            # get context token embeddings and predict
            ctx_tokens = self.ctx_encoder(x2, mask=ctx_mask)
            prd_tokens = self.predictor(ctx_tokens, ctx_mask, tgt_mask)

            # get target token embeddings
            with torch.no_grad():
                # embed full batch without grads
                tgt_tokens = self.tgt_encoder(x1)
                # keep only tokens in target block
                tgt_tokens = masks.gather_tokens(tgt_tokens, tgt_mask) 
                # if self.cfg.norm_target: # TODO: is norm any help?
                #     target = self.norm(target)

            # similarity loss
            loss += -self.sim(prd_tokens, tgt_tokens)
        
        return loss
    
    
    def update(self, optimizer, loss, step=None, total_steps=None):
        
        # student update
        super().update(optimizer, loss)

        # teacher update via exponential moving average of student
        tau = self.cfg.ema_momentum
        if self.cfg.momentum_schedule: # linear increase to tau=1
            frac = step/total_steps
            tau = tau + (1-tau)*frac

        for ps, pt in zip(self.ctx_encoder.parameters(), self.tgt_encoder.parameters()):
            pt = tau*pt + (1-tau)*ps

    def forward(self, x, mask=False):
        return self.ctx_encoder(x, mask=mask)

    @torch.inference_mode()
    def embed(self, x):
        return self.ctx_encoder(x)