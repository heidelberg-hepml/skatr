import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset

class LabelledDataset(Dataset):

    def __init__(self, cfg, device, summary_net=None):
        
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.Xs, self.ys = [], []

        for f in self.files:
            record = np.load(f)
            X = torch.from_numpy(record['image']).to(torch.get_default_dtype()) # TODO: Add option for `channels_last` memory format?
            y = torch.from_numpy(record['label']).to(torch.get_default_dtype()) # TODO: Cast with numpy before
            self.Xs.append(X)
            self.ys.append(y)
            if cfg.on_gpu:
                X = X.to(device)
                y = y.to(device)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]
    

class LabelledDatasetByFile(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
        y = torch.from_numpy(record['label']).to(torch.get_default_dtype())

        return X, y


class UnlabelledDataset(Dataset):

    def __init__(self, cfg, device):
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.Xs = []
        
        for f in self.files:
            record = np.load(f)
            X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
            if cfg.on_gpu:
                X = X.to(device)
            self.Xs.append(X)


class UnlabelledDatasetByFile(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
        return X,


# class SummarizedDataset(Dataset):

#     def __init__(self, cfg, device):
#         raise NotImplementedError
#         # self.h5_file = ...
#         # self.Xs = 
#         # self.ys = 

#     def __len__(self):
#         return len(self.Xs)

#     def __getitem__(self, idx):
#         return self.Xs[idx], self.ys[idx]        