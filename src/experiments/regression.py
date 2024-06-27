import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from glob import glob
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset

from src.experiments.base_experiment import BaseExperiment
from src.models import Regressor
from src.utils import PARAM_NAMES

class RegressionExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return RegressionDatasetByFile(self.cfg.data)
        else:
            return RegressionDataset(self.cfg.data, self.device)

    def get_model(self):
        return Regressor(self.cfg)
    
    def plot(self):
        
        # pyplot config
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble']=(
            r'\usepackage{amsmath}'
            r'\usepackage[bitstream-charter]{mathdesign}'
        )

        label_pred_pairs = np.load(os.path.join(self.exp_dir, 'label_pred_pairs.npy'))
        
        # check for existing plots
        savename = 'recovery.pdf'
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, 'old_plots')
            self.log.info(f'Moving old plots to {old_dir}')
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # marker settings
        ref_kwargs = {'color': 'crimson', 'ls': '--', 'lw': 2, 'alpha': 0.8}
        dot_kwargs = {'alpha': 0.6, 'color': 'navy', 's': 15}

        # create plots
        with PdfPages(savepath) as pdf:

            # iterate over individual parameters
            for i in range(label_pred_pairs.shape[1]):

                # make figure with ratio axis
                fig, ax = plt.subplots(figsize=(5,5))
                main_cell, ratio_cell = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=ax, height_ratios=[5,1.5], hspace=0.05
                )
                main_ax = plt.subplot(main_cell)
                ratio_ax = plt.subplot(ratio_cell)

                # unpack labels/preds and calculate metric
                labels, preds = label_pred_pairs[:, i].T
                lo, hi = labels.min(), labels.max() # range of true targets
                MARE = (abs(preds-labels)/labels).mean()
                
                # fill main axis
                pad = 0.02*(hi-lo)
                main_ax.scatter(labels, preds, **dot_kwargs)
                main_ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], **ref_kwargs)
                main_ax.text(0.1, 0.9, f"{MARE=:.1e}", transform=ax.transAxes)
                
                # fill ratio axis
                ratio_ax.scatter(labels, abs(preds - labels)/labels, **dot_kwargs)
                ratio_ax.semilogy()

                # axis labels
                main_ax.set_title(PARAM_NAMES[i], fontsize=14)
                main_ax.set_ylabel('Network', fontsize=13)
                ratio_ax.set_ylabel(
                    r'$\left|\frac{\text{Net}\,-\,\text{True}}{\text{True}}\right|$', fontsize=10
                )
                ratio_ax.set_xlabel('Truth', fontsize=13)

                # axis limits
                main_ax.set_xlim([lo-pad, hi+pad])
                main_ax.set_ylim([lo-pad, hi+pad])
                ratio_ax.set_xlim(*main_ax.get_xlim())
                ratio_ax.set_ylim(1e-3, 1)

                # clean
                ax.set_axis_off()
                main_ax.set_xticklabels([])

                pdf.savefig(fig, bbox_inches='tight')

        self.log.info(f'Saved plots to {savepath}')
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        """
        Evaluates the regressor on lightcones in the test dataset.
        Predictions are saved alongside truth labels
        """
        
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
        savepath = os.path.join(self.exp_dir, 'label_pred_pairs.npy')
        self.log.info(f'Saving label/prediction pairs to {savepath}')
        np.save(savepath, np.stack([labels, preds], axis=-1))


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