import torch # for rot90 augmentations
import random
from omegaconf import DictConfig

from .base_model import Model
from .. import networks


class SimSiam(Model):

    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.predictor = networks.MLP(cfg.predictor)
        self.aug_num = cfg.training.aug_num
        self.aug_type = cfg.training.aug_type

    def batch_loss(self, batch):
    #xx what is batch.shape
    # TODO: check functionality

        def augment(x):
            def random_ints(max):
                rand_arr = random.sample(range(0, max+1), 2)
                return rand_arr[0], rand_arr[1]

            def random_int(max):
                return random.randint(0, max)

            def rotate(x1, x2):
                i1, i2 = random_ints(3) # random integers [0, 3]
                x1 = torch.rot90(x1, i1, dims=[2,3]) # rotate between [0, 3] times
                x2 = torch.rot90(x2, i2, dims=[2,3])
                return x1, x2
            
            def reflect(x1, x2):
                i = random_int(3)
                if i == 1:
                    x1 = x1.transpose(2, 3)
                if i == 2:
                    x2 = x2.transpose(2, 3)
                if i == 3:
                    x1 = x1.transpose(2, 3)
                    x2 = x2.transpose(2, 3)
                return x1, x2
            
            # apply augmentations
            x1, x2 = x, x
            if 'rotation' in self.aug_type:
                x1, x2 = rotate(x1, x2)
            if 'reflection' in self.aug_type:
                x1, x2 = reflect(x1, x2)
                
            # check if same augmentation was applied
            diff = torch.sum(x1[2]-x2[2]) + torch.sum(x1[3]-x2[3])
            heat = torch.sum(x1[2])
            if abs(diff) < 1.e-11:
                if abs(heat) > 1.e-8:
                    del x1, x2
                    return augment(x)

            return x1, x2

        def cosSim(p, z):
            # TODO eps eps=1e-6
            CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            z = z.detach() # stop gradient
            return CosineSimilarity(p, z)
        
        loss = 0
        for i in range(self.aug_num):
            # TODO dipose of [0]
            x = batch[0]
            # augment / mask batch
            x1, x2 = augment(x)

            # TODO masking
            #z1 = self(x1, masking=True)

            # embed original and transformed batch
            z1 = self(x1)
            z2 = self(x2)

            # predict original from embedding of transformed
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

            # symmetric loss using stopgrad on z
            loss += - 0.5 * ( cosSim(p1, z2) + cosSim(p2, z1) )
        
        loss = torch.mean(loss)/self.aug_num
        return loss
    
    def forward(self, x):
        return self.net(x)
        #raise NotImplementedError

    def predict(self, x):
        return self.net(x)
        #raise NotImplementedError