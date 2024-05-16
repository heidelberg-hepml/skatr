from .base_model import Model

class SimSiam(Model):

    def batch_loss(self, batch):
        # augment / mask batch
        # embed original and transformed batch
        # predict original from embedding of transformed

        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError