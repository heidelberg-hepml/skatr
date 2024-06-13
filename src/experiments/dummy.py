import torch
from torch.utils.data import TensorDataset

from src.experiments.base_experiment import BaseExperiment
from src.models import *

class DummyExperiment(BaseExperiment):
    
    def get_dataset(self):
        dataset = TensorDataset(
            torch.rand((20, 1, 140, 140, 2350)).to(memory_format=torch.channels_last_3d),
            torch.rand((20, 6))
        )
        return dataset

    def get_model(self):
        return Regressor(self.cfg)
    
    def plot(self):
        pass
    
    def evaluate(self, dataloaders):
        pass