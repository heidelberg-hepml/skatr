import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src import networks
from src.models.base_model import Model
from src.utils import augmentations, masks

class Pretrainer(Model):

    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.predictor = networks.MLP(cfg.predictor)
        self.student = self.net
        self.teacher = self.net.__class__(cfg.net)
        self.norm = nn.BatchNorm1d(cfg.latent_dim)
        self.augment = augmentations.RotateAndReflect()

        if cfg.sim=='cosine':
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        elif cfg.sim=='l2':
            self.sim = lambda x1, x2: -F.mse_loss(x1, x2)
        elif cfg.sim=='l1':
            self.sim = lambda x1, x2: -F.l1_loss(x1, x2)

    def batch_loss(self, batch):        

        # augment batch
        x1 = batch[0]
        x2 = self.augment(x1) if self.cfg.augment else x1

        # sample mask
        mask = self.sample_mask(x1.size(0), x1.device)
            
        # embed masked batch
        embedding = self.student(x1, mask=mask)

        # embed full batch without grads
        with torch.no_grad():
            target = self.teacher(x2)
            if self.cfg.norm_target:
                target = self.norm(target)

        # predict teacher embedding from student embedding
        pred = self.predictor(embedding)

        # similarity loss
        loss = -self.sim(pred, target)
        
        return loss.mean()
    
    def update(self, optimizer, loss, step=None, total_steps=None):
        
        # student update
        super().update(optimizer, loss)

        # teacher update via exponential moving average of student
        tau = self.cfg.ema_momentum
        if self.cfg.momentum_schedule: # linear increase to tau=1
            frac = step/total_steps
            tau = tau + (1-tau)*frac

        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt = tau*pt + (1-tau)*ps

    def forward(self, x, mask=None):
        return self.student(x, mask=mask)

    @torch.inference_mode()
    def embed(self, x):
        return self.student(x)
    
    def sample_mask(self, batch_size, device):

        if cfg := self.cfg.masking:
            num_patches = self.student.num_patches
            match cfg.name:
                case 'random':
                    return masks.random_patch_mask(num_patches, cfg, batch_size, device)