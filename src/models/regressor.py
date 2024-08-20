import torch
import torch.nn.functional as F

from src.models.base_model import Model

class Regressor(Model):

    def batch_loss(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y, y_pred)
        return loss
    
    def forward(self, x):
        if self.cfg.backbone and not self.cfg.frozen_backbone:
            with torch.no_grad():
                x = self.bb(x)
                if not hasattr(self.bb, 'head'):
                    x = x.mean(1) # (B, T, D) --> (B, D)

        return self.net(x)

    @torch.inference_mode()
    def predict(self, x):
        return self.forward(x)