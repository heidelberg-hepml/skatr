import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os
import torch
from getdist import plots, MCSamples
from glob import glob
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset

from src.experiments.base_experiment import BaseExperiment
from src.models import ConditionalFlowMatcher
from src.utils import PARAM_NAMES

class InferenceExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return InferenceDatasetByFile(self.cfg.data)
        else:
            return InferenceDataset(self.cfg.data, self.device)

    def get_model(self):
        return ConditionalFlowMatcher(self.cfg)
    
    def plot(self):
        """Adapted from https://github.com/heidelberg-hepml/21cm-cINN/blob/main/Plotting.py"""
        # load data
        record = np.load(os.path.join(self.exp_dir, 'param_posterior_pairs.npz'))
        params, samples = record['params'], record['samples']

        # check for existing plots
        savename = 'posteriors.pdf'
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, 'old_plots')
            self.log.info(f'Moving old plots to {old_dir}')
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # create plots
        with PdfPages(savepath) as pdf:

            # iterate test poitns
            for j in range(len(samples)):
                samp_mc = MCSamples(
                    samples=samples[j],
                    names=PARAM_NAMES,
                    labels=[l.replace('$', '') for l in PARAM_NAMES]
                )
                g = plots.get_subplot_plotter()
                g.settings.legend_fontsize = 18
                g.settings.axes_fontsize=18
                g.settings.axes_labelsize=18
                g.settings.linewidth=2
                g.settings.line_labels=False
                colour=['orchid']
                g.triangle_plot(
                    [samp_mc], filled=True, legend_loc='upper right',
                    colors=colour, contour_colors=colour
                )
                # add truth to 1d and 2d marginals
                for i in range(6):
                    ax = g.subplots[i,i].axes
                    ax.axvline(params[j,i], color='k', ls='--',lw=2)
                for n, m in combinations(range(6), 2):
                    ax = g.subplots[m,n].axes
                    ax.scatter(params[j,n],params[j,m],color='k',marker='x',s=100)
                
                post_patch = mpatches.Patch(color=colour[0], label='Posterior')
                true_line = mlines.Line2D(
                    [], [], color='k', marker='x',ls='--',lw=2, markersize=10, label='True'
                )
                g.fig.legend(
                    handles=[post_patch,true_line], bbox_to_anchor=(0.98, 0.98), fontsize=14
                )
                pdf.savefig(g.fig)
                
        self.log.info(f'Saved plots to {savepath}')      
    
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
            print(f'Sampling posterior for test point {i+1}')
            lc_batch = lcs[i].unsqueeze(0).repeat(self.cfg.sample_batch_size, 1, 1, 1, 1)
            posterior_samples.append(torch.vstack([
                model.sample_batch(lc_batch)
                for _ in range(self.cfg.num_posterior_samples//self.cfg.sample_batch_size)
            ]))
        
        # stack samples and postprocess
        posterior_samples = torch.stack(posterior_samples)
        for transform in reversed(self.preprocessing['y']):
            posterior_samples = transform.reverse(posterior_samples)

        # save results
        savepath = os.path.join(self.exp_dir, 'param_posterior_pairs.npz')
        self.log.info(f'Saving parameter/posterior pairs to {savepath}')
        np.savez(
            savepath,
            params=params[:self.cfg.num_test_points].numpy(),
            samples=posterior_samples.numpy()
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