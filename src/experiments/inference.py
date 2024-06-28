import matplotlib.pyplot as plt
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
from src.models import ConditionalFlowMatcher, INN
from src.utils.plotting import PARAM_NAMES

class InferenceExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return InferenceDatasetByFile(self.cfg.data)
        else:
            return InferenceDataset(self.cfg.data, self.device)

    def get_model(self):
        if self.cfg.generative_model == 'CFM':
            return ConditionalFlowMatcher(self.cfg)
        elif self.cfg.generative_model == 'INN':
            return INN(self.cfg)        
    
    def plot(self):
        """Adapted from https://github.com/heidelberg-hepml/21cm-cINN/blob/main/Plotting.py"""
        # load data
        record = np.load(os.path.join(self.exp_dir, 'param_posterior_pairs.npz'))
        params  = record['params'] 
        samples = record['samples']

        # posterior
        # check for existing plot
        savename = 'posteriors.pdf'
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, 'old_plots')
            self.log.info(f"Moving old '{savename}' to {old_dir}")
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # create plots
        with PdfPages(savepath) as pdf:

            # iterate test poitns
            for j in range(min(len(samples), 8)):
                samp_mc = MCSamples(
                    samples=samples[j], names=PARAM_NAMES,
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
                plt.close()

        self.log.info(f"Saved posterior plots as '{savename}'")

        # calibration
        param_logprobs = record['param_logprobs']
        sample_logprobs = record['sample_logprobs']
        
        fs = [
            (t > p).mean() for t, p in zip(param_logprobs, sample_logprobs)
        ]

        # TARP calibration
        # mins, maxs = params.min(0), params.max(0)
        # ref_params = torch.rand_like(params) * (maxs-mins) + mins
        # tarps = [
        #     ((t-r).abs() > (p-r).abs()).mean() for r, t, p in zip(ref_params, params, samples)
        # ]

        savename='calibration.pdf'
        bins = np.linspace(0, 1, 20)
        fig, ax = plt.subplots(figsize=(4,4), dpi=200)
        ax.plot([0,1], [0,1], ls='--', color='darkgray')
        ax.plot(bins, np.quantile(fs, bins), color='crimson')
        ax.set_ylabel(r'Quantile', fontsize=13)
        ax.set_xlabel('$f$', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(self.exp_dir, savename))
        self.log.info(f"Saved calibration plot as '{savename}'")

    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        """
        Generates samples from the posterior distributions for a select
        number of lightcones from the test set. The samples are saved
        alongside truth parameter values.
        """

        # disable batchnorm updates, dropout etc.
        model.eval()
        lc_batch, params = next(iter(dataloaders['test']))
        posterior_samples, posterior_logprobs = [], []
        
        # preprocess lightcones and parameters
        lc_batch = lc_batch[:self.cfg.num_test_points].to(self.device)
        processed_params = params[:self.cfg.num_test_points].to(self.device)
        for transform in self.preprocessing['x']:
            lc_batch = transform.forward(lc_batch)
        for transform in self.preprocessing['y']:
            processed_params = transform.forward(processed_params)

        # evaluate test param likelihoods
        param_logprobs = model.log_prob(processed_params, model.sum_net(lc_batch)).cpu()

        # loop over test points
        for i in range(self.cfg.num_test_points):
            self.log.info(f'Sampling posterior for test point {i+1}')
            
            # select corresponding lightcone
            lc = lc_batch[i].unsqueeze(0).repeat(self.cfg.sample_batch_size, 1, 1, 1, 1)

            # sample posterior in batches
            sample_list, logprob_list = [], []
            for _ in range(self.cfg.num_posterior_samples//self.cfg.sample_batch_size):
                sample, logprob = model.sample_batch(lc)
                sample_list.append(sample.detach().cpu())
                logprob_list.append(logprob.detach().cpu())
            
            # collect samples
            posterior_samples.append(torch.vstack(sample_list))
            posterior_logprobs.append(torch.vstack(logprob_list))

            del lc

        # stack all samples
        posterior_samples = torch.stack(posterior_samples)
        posterior_logprobs = torch.stack(posterior_logprobs)
        
        # postprocess posterior samples
        for transform in reversed(self.preprocessing['y']):
            posterior_samples = transform.reverse(posterior_samples)

        # save results
        savename = 'param_posterior_pairs.npz'
        savepath = os.path.join(self.exp_dir, savename)
        self.log.info(f'Saving parameter/posterior pairs as {savename}')
        np.savez(
            savepath,
            params=params[:self.cfg.num_test_points].numpy(),
            param_logprobs=param_logprobs.numpy(),
            samples=posterior_samples.numpy(),
            sample_logprobs=posterior_logprobs.numpy()
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