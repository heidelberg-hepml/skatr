import torch
import torch.nn.functional as F

from .base_model import Model

class Regressor(Model):

    def batch_loss(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y, y_pred)
        return loss
    
    def forward(self, x):
        if self.cfg.backbone:
            with torch.no_grad():
                x = self.bb(x)
        return self.net(x)

    @torch.inference_mode()
    def predict(self, x):
        if self.cfg.backbone:
            x = self.bb(x)
        return self.net(x)