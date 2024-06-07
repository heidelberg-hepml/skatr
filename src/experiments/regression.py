import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset, TensorDataset

from .base_experiment import BaseExperiment
from ..models import Regressor

class RegressionExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return RegressionDatasetByFile(self.cfg.data)
        else:
            return RegressionDataset(self.cfg.data, self.device)

    def get_model(self):
        return Regressor(self.cfg)
    
    def plot(self):
        label_pred_pairs = np.load(os.path.join(self.exp_dir, 'label_pred_pairs.npy'))
        
        # check for existing plots
        savename = 'recovery.pdf'
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, 'old_plots')
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # create plots
        with PdfPages(savepath) as pdf:

            # iterate over individual parameters
            for i in range(label_pred_pairs.shape[1]):

                fig, ax = plt.subplots(figsize=(5,4))
                labels, preds = label_pred_pairs[:, i].T

                lo, hi = labels.min(), labels.max() # range of true targets
                NRMSE = np.sqrt(((labels-preds)**2).mean())/(hi-lo)
                
                pad = 0.1*(hi-lo)
                ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color='crimson', ls='--', lw=2)
                ax.scatter(labels, preds, alpha=0.4, color='darkblue')
                ax.text(0.1, 0.9, f"{NRMSE=:.3f}", transform=ax.transAxes)

                ax.set_xlabel(f'Param {i}')
                ax.set_ylabel(f'Prediction')
                fig.tight_layout()

                pdf.savefig(fig)
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):

        # disable batchnorm updates, dropout etc.
        model.eval()

        # get truth targets and predictions across the test set
        labels, preds = [], []
        for x, y in dataloaders['test']:

            labels.append(y.numpy())
            
            # preprocess input
            x = x.to(self.device)
            for transform in self.preprocessing['x']:
                x = transform.forward(x)

            # predict
            pred = model.predict(x).detach().cpu()

            # postprocess output
            for transform in reversed(self.preprocessing['y']):
                pred = transform.reverse(pred)
            
            # append prediction
            preds.append(pred.numpy())

        # stack results
        labels = np.vstack(labels)
        preds = np.vstack(preds)
        
        # save results
        np.save(
            os.path.join(self.exp_dir, 'label_pred_pairs'), np.stack([labels, preds], axis=-1)
        )


class RegressionDatasetByFile(Dataset):

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

class RegressionDataset(Dataset):

    def __init__(self, cfg, device):
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