import torch.nn.functional as F

from src.models.base_model import Model

class Classifier(Model):

    def batch_loss(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss
    
    def forward(self, x):
        """Return class logits"""
        return self.net(x)

    def predict(self, x):
        """Return class probabilities"""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def classify(self, x):
        raise NotImplementedError
        """Return index of most likely class"""