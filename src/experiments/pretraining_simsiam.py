import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset

from .base_experiment import BaseExperiment
from ..models import SimSiam

class SimSiamExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return SimSiamDatasetByFile(self.cfg.data)
        else:
            return SimSiamDataset(self.cfg.data)

    def get_model(self):
        return SimSiam(self.cfg)
    
    def plot(self):
        raise NotImplementedError
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        raise NotImplementedError


class SimSiamDatasetByFile(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
        return X, 


class SimSiamDataset(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.Xs = []
        
        for f in self.files:
            record = np.load(f)
            X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
            self.Xs.append(X)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], 