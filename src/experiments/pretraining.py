import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset

from src.experiments.base_experiment import BaseExperiment
from src.models import Pretrainer

class PretrainingExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return PretrainingDatasetByFile(self.cfg.data)
        else:
            return PretrainingDataset(self.cfg.data, self.device)

    def get_model(self):
        return Pretrainer(self.cfg)
    
    def plot(self):
        raise NotImplementedError
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        raise NotImplementedError


class PretrainingDatasetByFile(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
        return X, 


class PretrainingDataset(Dataset):

    def __init__(self, cfg, device):
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.Xs = []
        
        for f in self.files:
            record = np.load(f)
            X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
            if cfg.on_gpu:
                X = X.to(device)
            self.Xs.append(X)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx],