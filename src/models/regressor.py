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
        if self.cfg.backbone and not self.cfg.data.summarize and not self.cfg.replace_backbone:
            x = self.bb(x)
            # if not hasattr(self.bb, 'head') and self.net.cfg.arch == 'MLP': # weird...
            if not hasattr(self.bb, 'head') and self.cfg.net.arch == 'MLP': #TODO: Clean
                x = x.mean(1) # (B, T, D) --> (B, D)

        return self.net(x)

    @torch.inference_mode()
    def predict(self, x):
        return self.forward(x)