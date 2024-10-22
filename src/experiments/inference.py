import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import numpy as np
import os
import random
import torch
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from src import models
from src.experiments.base_experiment import BaseExperiment
from src.utils import datasets
from src.utils.plotting import PARAM_NAMES


class InferenceExperiment(BaseExperiment):

    def get_dataset(self, directory):
        prep = self.preprocessing
        if self.cfg.data.file_by_file:
            return datasets.LCDatasetByFile(
                self.cfg.data, directory, preprocessing=prep
            )
        else:
            return datasets.LCDataset(
                self.cfg.data, directory, self.device, preprocessing=prep
            )

    def get_model(self):

        try:
            model_cls = getattr(models, self.cfg.generative_model)
        except AttributeError as e:
            self.log.error(f"Unknown model class {self.cfg.generative_model}.")
            raise e

        return model_cls(self.cfg)

    @torch.inference_mode()
    def evaluate(self, dataloaders):
        """
        Generates samples from the posterior distributions for a select
        number of lightcones from the test set. The samples are saved
        alongside truth parameter values.
        """

        assert self.cfg.num_test_points <= self.cfg.training.test_batch_size

        # initialize containers
        posterior_samples, posterior_logprobs, param_logprobs = [], [], []

        # disable batchnorm updates, dropout etc.
        self.model.eval()

        # pull data from the test set
        test_lcs, params = next(iter(dataloaders["test"]))
        test_lcs = test_lcs[: self.cfg.num_test_points]
        params = params[: self.cfg.num_test_points].to(self.device, self.dtype_train)

        # loop over test lcs in batches
        for j, (lc_batch, param_batch) in enumerate(
            zip(
                DataLoader(test_lcs, self.cfg.sample_batch_size),
                DataLoader(params, self.cfg.sample_batch_size),
            )
        ):

            # move batch to gpu
            lc_batch = lc_batch.to(self.device, self.dtype_train)

            # summarize
            lc_batch = self.model.summarize(lc_batch)

            # evaluate true param likelihoods
            param_logprobs.append(self.model.inn.log_prob(param_batch, lc_batch).cpu())

            # loop over test points
            for i in range(len(lc_batch)):
                self.log.info(
                    f"Sampling posterior for test point {j*self.cfg.sample_batch_size+i+1}"
                )

                # select corresponding lightcone
                lc = lc_batch[i].unsqueeze(0)
                lc = lc.repeat(self.cfg.sample_batch_size, *[1] * (lc.ndim - 1))

                # sample posterior in batches
                sample_list, logprob_list = [], []
                for _ in range(
                    self.cfg.num_posterior_samples // self.cfg.sample_batch_size
                ):
                    sample, logprob = self.model.inn.sample_batch(lc)
                    sample_list.append(sample.detach().cpu())
                    logprob_list.append(logprob.detach().cpu())

                # collect samples
                posterior_samples.append(torch.vstack(sample_list))
                posterior_logprobs.append(torch.vstack(logprob_list))

                del lc

        # stack containers into tensors
        posterior_samples = torch.stack(posterior_samples)
        posterior_logprobs = torch.stack(posterior_logprobs)
        param_logprobs = torch.hstack(param_logprobs)

        # postprocess
        params = params.cpu()
        for transform in reversed(self.preprocessing["y"]):
            posterior_samples = transform.reverse(posterior_samples)
            params = transform.reverse(params)

        # collect true parameters
        target_indices = sorted(self.cfg.target_indices)
        params = params[: self.cfg.num_test_points, target_indices]

        # save results
        savename = "param_posterior_pairs.npz"
        savepath = os.path.join(self.exp_dir, savename)
        self.log.info(f"Saving parameter/posterior pairs as {savename}")
        np.savez(
            savepath,
            params=params.numpy(),
            param_logprobs=param_logprobs.numpy(),
            samples=posterior_samples.numpy(),
            sample_logprobs=posterior_logprobs.numpy(),
        )

    def plot(self):
        # load data
        record = np.load(os.path.join(self.exp_dir, "param_posterior_pairs.npz"))
        samples = record["samples"]
        params = record["params"]
        colors = ["navy", "royalblue"]

        param_names = [PARAM_NAMES[i] for i in sorted(self.cfg.target_indices)]
        N = len(param_names)

        # posterior
        # check for existing plot
        savename = "posteriors.pdf"
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, "old_plots")
            self.log.info(f"Moving old '{savename}' to {old_dir}")
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # create plots
        with PdfPages(savepath) as pdf:

            for j in random.sample(range(N), min(N, 8)):

                fig = compare_posteriors(
                    [params[j]],
                    [samples[j]],
                    colors_list=[colors],
                    smooth=2.0,
                    bins=45,
                    param_names=PARAM_NAMES,
                )

                patch1 = mpatches.Patch(color=colors[0], label="Posterior", alpha=0.7)

                true_line = mlines.Line2D(
                    [],
                    [],
                    ls="--",
                    lw=2,
                    color="#181818",
                    marker="o",
                    markersize=6,
                    label="True",
                )
                fig.legend(
                    handles=[patch1, true_line],
                    bbox_to_anchor=(0.78, 0.62),
                    fontsize=14,
                    frameon=False,
                )

                pdf.savefig(fig, bbox_inches="tight", dpi=500)
                plt.show()
                plt.close()

        self.log.info(f"Saved posterior plots as '{savename}'")

        # calibration
        param_logprobs = record["param_logprobs"]
        sample_logprobs = record["sample_logprobs"]

        fs = [(t > p).mean() for t, p in zip(param_logprobs, sample_logprobs)]

        # TARP calibration
        # mins, maxs = params.min(0), params.max(0)
        # ref_params = torch.rand_like(params) * (maxs-mins) + mins
        # tarps = [
        #     ((t-r).abs() > (p-r).abs()).mean() for r, t, p in zip(ref_params, params, samples)
        # ]

        savename = "calibration.pdf"
        bins = np.linspace(0, 1, 20)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
        ax.plot([0, 1], [0, 1], ls="--", color="darkgray")
        ax.plot(bins, np.quantile(fs, bins), color="crimson")
        ax.set_ylabel(r"Quantile", fontsize=13)
        ax.set_xlabel("$f$", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(self.exp_dir, savename))
        self.log.info(f"Saved calibration plot as '{savename}'")


# plotting helpers
def add_1d_posterior(ax, idx, true, samples, bins=45, smooth=1.5, color="k"):

    # select current parameters
    true = true[idx]
    sample = samples.T[idx]

    # histogram the samples
    zs, edges = np.histogram(sample, bins=bins, density=True)

    centers = (edges[:-1] + edges[1:]) / 2

    # optional smoothing
    if smooth:
        zs = gaussian_filter(zs, smooth, mode="constant")

    # plot truth lines
    ax.plot(centers, zs, color=color, lw=1.0)
    ax.axvline(x=true, ls="--", color="#181818", alpha=0.8, lw=0.75)


def add_2d_posterior(ax, idcs, true, samples, bins=45, smooth=1.5, colors="k"):

    # select current parameters
    xs_samp, ys_samp = samples.T[idcs]

    # histogram the samples
    zs, xedges, yedges = np.histogram2d(xs_samp, ys_samp, bins=bins)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # normalize the histogram
    zs /= zs.sum()

    # optional smoothing
    if smooth:
        zs = gaussian_filter(zs, smooth, mode="constant")

    # calculate confidence interval levels
    zflat = zs.flatten()
    sort_idcs = np.argsort(-zflat)
    cum_prob = np.cumsum(zflat[sort_idcs])

    one_sigma_idx = np.argmin(abs(cum_prob - (1.0 - np.exp(-0.5 * 1**2))))
    two_sigma_idx = np.argmin(abs(cum_prob - (1.0 - np.exp(-0.5 * 2**2))))

    levels = zflat[sort_idcs][[two_sigma_idx, one_sigma_idx]]  # increasing

    # plot contours
    ax.contourf(
        *np.meshgrid(xcenters, ycenters, indexing="ij"),
        zs,
        levels=list(levels) + [1],
        colors=colors[::-1],
        linestyles=["dashed", "solid"],
        zorder=0,
        alpha=0.8,
    )

    # add truth lines
    add_truth(ax, idcs, true)


def add_truth(ax, idcs, true):
    x_true, y_true = true[idcs]
    ax.axhline(y=y_true, ls="--", color="#181818", alpha=0.8, lw=0.75)
    ax.axvline(x=x_true, ls="--", color="#181818", alpha=0.8, lw=0.75)
    ax.scatter(x_true, y_true, color="#181818", marker="o", s=10, zorder=1)


def compare_posteriors(
    params_list,
    samples_list,
    colors_list,
    bins=45,
    smooth=1.25,
    param_names=PARAM_NAMES,
):

    nparams = len(params_list[0].T)
    figsize = 1 + 10 / 6 * nparams
    fig, ax = plt.subplots(
        nparams,
        nparams,
        figsize=(figsize, figsize),
        dpi=400,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    for i, j in product(range(nparams), repeat=2):
        a = ax[i, j]

        if i < j:
            # no plot
            a.set_axis_off()
        elif i > j:
            for (
                params,
                samples,
                colors,
            ) in zip(params_list, samples_list, colors_list):
                # plot 2d posterior
                add_2d_posterior(
                    a, [j, i], params, samples, smooth=smooth, bins=bins, colors=colors
                )
        else:
            for (
                params,
                samples,
                colors,
            ) in zip(params_list, samples_list, colors_list):
                # plot 1d posterior
                add_1d_posterior(
                    a, i, params, samples, smooth=smooth, bins=bins, color=colors[0]
                )

        # clean ticks mid-figure
        if i < nparams - 1:
            a.set_xticks([])
        else:
            a.xaxis.set_major_locator(ticker.MaxNLocator(3))

        # if j > 0 and j!=i:
        if j > 0 or i == 0:
            a.set_yticks([])
        else:
            a.yaxis.set_major_locator(ticker.MaxNLocator(3))

        # parameter labels
        if j == 0 and i != 0:
            a.set_ylabel(param_names[i], fontsize=13, rotation=90)
        if i == nparams - 1:
            a.set_xlabel(param_names[j], fontsize=13)

    # align x axis of 1d dists
    for i in range(nparams):

        ax[i, i].set_ylim(0, None)
        if i < nparams - 1:
            ax[i, i].set_xlim(*ax[i + 1, i].get_xlim())

    return fig
