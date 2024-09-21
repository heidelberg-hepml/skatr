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
        self.novar_frac = cfg.novar_frac
        self.stopvar = int(bool(self.novar_frac))

    def batch_loss(self, batch):
        x, y = batch
        
        logit_mean, invsp_var = self(x)
        
        mean = F.sigmoid(logit_mean)
        var = F.softplus(invsp_var)

        # optionally disable variance
        # logvar = (1-self.stopvar) * logvar
        var = (1-self.stopvar) * var + self.stopvar / 100
        
        # gaussian likelihood
        # loss = (y - mean)**2 / (2*logvar.exp()) + 0.5*logvar
        loss = (y - mean)**2 / (2*var) + 0.5*var.log()
            
        return loss.mean()

    def forward(self, x):
        logit_mean, invsp_var = super().forward(x).tensor_split(2, dim=-1)
        return logit_mean, invsp_var

    @torch.inference_mode()
    def predict(self, x):
        logit_mean, invsp_var = self(x)
        mean = F.sigmoid(logit_mean)
        var = F.softplus(invsp_var)
        return mean, var

    def update(self, optimizer, loss, step=None, total_steps=None):
        
        # default update
        super().update(optimizer, loss)

        # enable variance
        if step/total_steps >= self.novar_frac:
            self.stopvar = 0.