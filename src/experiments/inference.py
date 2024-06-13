import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset, DataLoader, IterableDataset

from src.experiments.base_experiment import BaseExperiment
from src.models import ConditionalFlowMatcher

class InferenceExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return InferenceDatasetByFile(self.cfg.data)
        else:
            return InferenceDataset(self.cfg.data, self.device)

    def get_model(self):
        return ConditionalFlowMatcher(self.cfg)
    
    def plot(self):
        raise NotImplementedError
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        """
        Generates samples from the posterior distributions for a select
        number of lightcones from the test set. The samples are saved
        alongside truth parameter values.
        """
        
        # disable batchnorm updates, dropout etc.
        model.eval()
        posterior_samples = []
        lcs, params = next(iter(dataloaders['test']))
        
        # preprocess input
        lcs = lcs.to(self.device)
        for transform in self.preprocessing['x']:
            lcs = transform.forward(lcs)

        for i in range(self.cfg.num_test_points):
            lc_batch = lcs[i].unsqueeze(0).repeat(self.cfg.sample_batch_size, 1, 1, 1, 1)
            posterior_samples.append(np.vstack([
                model.sample_batch(lc_batch)
                for _ in range(self.cfg.num_posterior_samples//self.cfg.sample_batch_size)
            ]))
        
        # stack samples and postprocess
        posterior_samples = np.stack(posterior_samples)
        for transform in reversed(self.preprocessing['y']):
            posterior_samples = transform.reverse(posterior_samples)

        # save results
        np.savez(
            os.path.join(self.exp_dir, 'param_samples_pairs'),
            params=params[:self.cfg.num_test_points].numpy(),
            samples=posterior_samples,
        )

class InferenceDatasetByFile(Dataset):

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

class InferenceDataset(Dataset):

    def __init__(self, cfg, device):
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))
        self.Xs, self.ys = [], []
        
        for f in self.files:
            record = np.load(f)
            X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
            y = torch.from_numpy(record['label']).to(torch.get_default_dtype())
            self.Xs.append(X)
            self.ys.append(y)
            if cfg.on_gpu:
                X = X.to(device)
                y = y.to(device)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]

class IterDataset(IterableDataset):

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator    