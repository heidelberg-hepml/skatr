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
        self.norm = nn.BatchNorm1d(cfg.latent_dim)  # TODO: Remove norm?
        self.augment = augmentations.RotateAndReflect()

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
        ctx_mask, tgt_masks = masks.sample_jepa_masks(x2.size(0), x2.device) # TODO: Update parameter
            
        
        # embed masked batch
        ctx_tokens = masks.gather_tokens(x2, ctx_mask)
        ctx_tokens = self.ctx_encoder(ctx_tokens)

        loss = 0.
        for tgt_mask in tgt_masks:
            # predict tokens
            prd_tokens = self.predictor(ctx_tokens, ctx_mask, tgt_mask)
            with torch.no_grad():
                # embed full batch without grads
                tgt_tokens = self.tgt_encoder(x1)
                # if self.cfg.norm_target: # TODO: is norm any help?
                #     target = self.norm(target)

            # similarity loss
            loss += -self.sim(prd_tokens, tgt_tokens)
        
        return loss.mean() # TODO: Take the mean more carefully, in case each mask is different size
    
    
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