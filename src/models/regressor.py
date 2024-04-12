import torch.nn.functional as F

from .base_model import Model

class Regressor(Model):

    def batch_loss(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y, y_pred)
        return loss
    
    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.net(x)