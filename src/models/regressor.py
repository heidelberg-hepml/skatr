import torch
import torch.nn.functional as F

from src.models.base_model import Model

class Regressor(Model):

    def batch_loss(self, batch):
        x, y = batch
        y_pred = self(x)
        match self.cfg.loss:
            case 'l1':
                loss = F.l1_loss(y, y_pred)
            case 'l2': 
                loss = F.mse_loss(y, y_pred)
            case _:
                raise ValueError(f"Unknown loss {self.cfg.loss}")
            
        return loss
    
    def forward(self, x):
        
        if hasattr(self, 'summary_net') and not self.cfg.data.summarize:
        
            x = self.summary_net(x)
            # if not hasattr(self.bb, 'head') and self.net.cfg.arch == 'MLP': # weird...
            if not hasattr(self.summary_net, 'head') and self.cfg.net.arch == 'MLP': #TODO: Clean
                x = x.mean(1) # (B, T, D) --> (B, D)

        return self.net(x)

    @torch.inference_mode()
    def predict(self, x):
        return self.forward(x)

class GaussianRegressor(Regressor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.const_sigma_frac = cfg.const_sigma_frac
        self.stop_sigma = int(bool(self.const_sigma_frac))

    def batch_loss(self, batch):
        
        x, y = batch
        
        mu, sigma = self(x)
        # optionally fix sigma constant
        sigma = (1-self.stop_sigma) * sigma + self.stop_sigma
        # gaussian likelihood
        loss = 0.5 * ((y - mu) / sigma)**2 + sigma.log()
            
        return loss.mean()

    def forward(self, x):
        logit_mu, invsp_sig = super().forward(x).tensor_split(2, dim=-1)
        mu = F.sigmoid(logit_mu)
        sigma = F.softplus(invsp_sig)
        return mu, sigma

    @torch.inference_mode()
    def predict(self, x):
        return self(x)

    def update(self, optimizer, loss, step=None, total_steps=None):
        
        # default update
        super().update(optimizer, loss)

        # enable variance
        if step/total_steps >= self.const_sigma_frac:
            self.stop_sigma = 0.
