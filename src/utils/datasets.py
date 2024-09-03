import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader

from src.utils.augmentations import RotateAndReflect

class LCDataset(Dataset):

    def __init__(self, cfg, directory, device, use_labels=True, preprocessing=None):

        self.use_labels = use_labels
        self.files = sorted(glob(f'{directory}/run*.npz'))
        self.Xs, self.ys = [], []

        dtype = torch.get_default_dtype()

        for f in self.files:
            record = np.load(f)

            # read and preprocess
            X = torch.from_numpy(record['image']).to(dtype) # TODO: Add option for `channels_last` memory format?
            for f in preprocessing['x']:
                X = f.forward(X)
            self.Xs.append(X)
            
            if use_labels:
                # read and preprocess
                y = torch.from_numpy(record['label']).to(dtype) # TODO: Cast with numpy before
                for f in preprocessing['y']:
                    y = f.forward(y)
                self.ys.append(y)
            
            if cfg.on_gpu:
                X = X.to(device)
                if use_labels:
                    y = y.to(device)
        
    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return (self.Xs[idx], self.ys[idx]) if self.use_labels else (self.Xs[idx],)
    

class LCDatasetByFile(Dataset):

    def __init__(self, cfg, directory, use_labels=True, preprocessing=None):
        
        self.cfg = cfg
        self.use_labels = use_labels
        self.preprocessing = preprocessing

        self.files = sorted(glob(f'{directory}/run*.npz'))
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        
        # read
        X = torch.from_numpy(record['image']).to(self.dtype)
        # preprocess
        for f in self.preprocessing['x']:
            X = f.forward(X)
        
        if self.use_labels:
            # read
            y = torch.from_numpy(record['label']).to(self.dtype)
            # preprocess
            for f in self.preprocessing['y']:
                y = f.forward(y)

        return (X, y) if self.use_labels else (X,)
    

class SummarizedLCDataset(Dataset):

    def __init__(
        self, dataset, summary_net, device, augment=False, summary_batch_size=32, num_cpus=0
        ):
        
        self.Xs = []
        self.ys = []
        self.summary_net = summary_net

        if augment:
            aug = RotateAndReflect()

        dataloader = DataLoader(
            dataset, batch_size=summary_batch_size, num_workers=num_cpus if num_cpus > 1 else 0
        )
        for X, y in dataloader:
            
            X = X.to(device)
            self.Xs.append(self.summarize(X))
            self.ys.append(y)
            if augment:
                for x in aug.enumerate(X):
                    self.Xs.append(self.summarize(x))
                    self.ys.append(y)

        self.Xs = torch.vstack(self.Xs)
        self.ys = torch.vstack(self.ys)

    @torch.no_grad()
    def summarize(self, x):
        x = self.summary_net(x)
        if not hasattr(self.summary_net, 'head'):
            x = x.mean(1) # (B, T, D) --> (B, D)
        return x

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]