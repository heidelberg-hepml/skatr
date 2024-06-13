import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint

from src import networks
from src.models.base_model import Model

class ConditionalFlowMatcher(Model):

    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.sum_net = self.bb if cfg.backbone else (
            networks.ViT(cfg.summary_net)
        )
        self.t_embed = TimeEmbedding(cfg)

    def batch_loss(self, batch):
        # unpack batch. note that for inference, we're modelling
        # a dist on the params, not the lightcone.
        cond, x = batch

        # sample time and latent space
        device = x.device
        t = torch.rand([x.size(0), 1], device=device)
        x0 = torch.randn_like(x, device=device)

        # interpolate latent sample to batch
        xt = t*x + (1-t)*x0

        # calculate mse loss to target vector
        loss = F.mse_loss(self(xt, t, cond), x-x0)
        
        return loss
    
    def forward(self, x, t, cond):
        """Evaluate the vector field network, conditional on the summary vector"""
        sum_vec = self.sum_net(cond)
        t_vec = self.t_embed(t)
        inp = torch.cat([x, t_vec, sum_vec], dim=1)
        return self.net(inp)

    @torch.inference_mode()
    def sample_batch(self, cond):
        """Draw conditional samples by solving the ODE"""
        
        device = cond.device
        batch_size = cond.size(0)

        # define ode solve function
        def solve_fn(t, x):
            t = t.repeat(batch_size).unsqueeze(1).to(device)
            return self(x, t, cond)

        # solve ode
        x0 = torch.randn((batch_size, self.cfg.dim)).to(device)
        solution_times = torch.tensor([0., 1.]).to(device)
        with torch.inference_mode():
            sample = odeint(
                solve_fn, x0, solution_times,
                **OmegaConf.to_object(self.cfg.solver_kwargs)
            )[-1]

        return sample.detach().cpu()

    
class TimeEmbedding(nn.Module):

    def __init__(self, cfg:DictConfig):
        super().__init__()
        dim = cfg.time_embed_dim
        self.register_buffer('freqs', torch.randn(dim//2))
        self.linear = nn.Linear(dim, dim)

    def forward(self, t):
        w = (self.freqs * 2 * math.pi).unsqueeze(0)
        t = w*t
        t = torch.cat([t.sin(), t.cos()], dim=1)
        return self.linear(t)




