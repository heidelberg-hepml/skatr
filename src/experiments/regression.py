import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset, TensorDataset

from .base_experiment import BaseExperiment
from ..models import *

class RegressionExperiment(BaseExperiment):
    
    def get_dataset(self):
        return LCRegressionDataset(self.cfg.data)

    def get_model(self):
        return Regressor(self.cfg)
    
    def plot(self):
        label_pred_pairs = np.load(os.path.join(self.exp_dir, 'label_pred_pairs.npy'))        
        with PdfPages('calibration.pdf') as pdf:

            # iterate over individual parameters
            for i in range(label_pred_pairs.shape[1]):

                fig, ax = plt.subplots(figsize=(5,4))
                pairs = label_pred_pairs[:, i]
                lo, hi = pairs[:, 0].min(), pairs[:, 0].max() # range of true targets
                pad = 0.1*(hi-lo)
                ax.plot([lo-pad]*2, [hi+pad]*2, color='gray', ls='--')
                ax.scatter(pairs.T, alpha=0.6)
                
                ax.set_xlabel(f'Param {i}')
                ax.set_ylabel(f'Prediction')
                fig.tight_layout()

                pdf.savefig(fig)
    
    def evaluate(self, dataloaders, model):

        # disable batchnorm updates, dropout etc.
        model.eval()

        # get truth targets and predictions across the test set
        labels, preds = [], []
        for X, y in dataloaders['test']:

            labels.append(y.numpy())
            with torch.inference_mode():
                
                # preprocess input
                X = X.to(self.device)
                _ = torch.empty_like(y).to(self.device)
                for transform in self.preprocessing:
                    X, _ = transform.forward(X, _)

                # predict
                pred = model.predict(X).detach().cpu()

                # postprocess output
                for transform in reversed(self.preprocessing):
                    _, pred = transform.reverse(torch.empty_like(X), pred)
                
                # append prediction
                preds.append(pred.numpy())

        # stack results
        labels = np.vstack(labels)
        preds = np.vstack(preds)
        
        # save results
        np.save(
            os.path.join(self.exp_dir, 'label_pred_pairs'), np.stack([labels, preds], axis=-1)
        )


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
            