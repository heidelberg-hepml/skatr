import numpy as np
import os
import torch
from glob import glob
from torch.utils.data import Dataset, TensorDataset

from .base_experiment import BaseExperiment
from .. import transforms
from ..models import *

class RegressionExperiment(BaseExperiment):
    
    def get_dataset(self):
        return LCRegressionDataset(self.cfg.data)

    def get_model(self):
        return Regressor(self.cfg)
    
    def plot(self):
        label_pred_pairs = np.load(os.path.join(self.exp_dir, 'label_pred_pairs'))
    
    def evaluate(self, dataloaders, model):

        # disable batchnorm updates, dropout etc.
        self.model.eval()

        # get truth targets and predictions across the test set
        labels, preds = [], []
        for X, y in dataloaders['test']:
            labels.append(y.numpy())
            with torch.inference_mode():
                preds.append(self.model.predict(X).detach().cpu().numpy())
        labels = np.vstack(labels)
        preds = np.vstack(preds)

        # calculate metrics
        mse = np.mean((labels - preds)**2)

        # save results
        np.save(os.path.join(self.exp_dir, 'label_pred_pairs'), np.stack([labels, preds], axis=-1))


class LCRegressionDataset(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(glob(f'{cfg.dir}/run*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype()) # TODO: Add option for `channels_last` memory format?
        y = torch.from_numpy(record['label']).to(torch.get_default_dtype()) # TODO: Cast with numpy before

        return X, y
            