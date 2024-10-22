import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from src.experiments.base_experiment import BaseExperiment
from src.models import Regressor, GaussianRegressor
from src.utils import datasets
from src.utils.plotting import PARAM_NAMES


class RegressionExperiment(BaseExperiment):

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
        return (GaussianRegressor if self.cfg.gaussian else Regressor)(self.cfg)


    @torch.inference_mode()
    def evaluate(self, dataloaders):
        """
        Evaluates the regressor on lightcones in the test dataset.
        Predictions are saved alongside truth labels
        """

        # disable batchnorm updates, dropout etc.
        self.model.eval()

        # get truth targets and predictions across the test set
        labels, preds, stds = [], [], []
        for x, y in dataloaders["test"]:

            # predict
            x = x.to(self.device, self.dtype_train)

            if self.cfg.gaussian:
                pred, std = self.model.predict(x)
                pred = pred.detach().cpu()
                std = std.detach().cpu()
            else:
                pred = self.model.predict(x).detach().cpu()

            # postprocess output
            for transform in reversed(self.preprocessing["y"]):
                pred = transform.reverse(pred)
                y = transform.reverse(y)

            # append prediction
            preds.append(pred.numpy())
            labels.append(y.cpu().numpy())
            if self.cfg.gaussian:
                stds.append(std.numpy())

        # stack results
        labels = np.vstack(labels)
        preds = np.vstack(preds)
        if self.cfg.gaussian:
            stds = np.vstack(stds)

        # save results
        savearrs = [labels, preds]
        if self.cfg.gaussian:
            savearrs.append(stds)

            for a in savearrs:
                print(a.shape)

        savepath = os.path.join(self.exp_dir, "label_pred_pairs.npy")
        self.log.info(f"Saving label/prediction pairs to {savepath}")
        np.save(savepath, np.stack(savearrs, axis=-1))


    def plot(self):

        # pyplot config
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}" r"\usepackage[bitstream-charter]{mathdesign}"
        )

        label_pred_pairs = np.load(os.path.join(self.exp_dir, "label_pred_pairs.npy"))

        # check for existing plots
        savename = "recovery.pdf"
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, "old_plots")
            self.log.info(f"Moving old plots to {old_dir}")
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # marker settings
        ref_kwargs = {"color": "#171717", "ls": "-", "lw": 0.5, "alpha": 0.8}
        err_kwargs = {
            "fmt": "o",
            "color": "navy",
            "ecolor": "royalblue",
            "ms": 2,
            "elinewidth": 1,
        }

        # create plots
        with PdfPages(savepath) as pdf:

            # iterate over individual parameters
            for i in range(label_pred_pairs.shape[1]):

                # make figure with ratio axis
                fig = plt.figure(figsize=(3.3, 3.5), constrained_layout=True)
                grid = gridspec.GridSpec(
                    2, 1, figure=fig, height_ratios=[5, 1.5], hspace=0.05
                )
                main_ax = plt.subplot(grid[0])
                ratio_ax = plt.subplot(grid[1])

                # unpack labels/preds and calculate metric
                if self.cfg.gaussian:
                    labels, preds, sigmas = label_pred_pairs[:, i].T
                else:
                    labels, preds = label_pred_pairs[:, i].T

                # digitize data
                num_bins = 40
                lo, hi = labels.min(), labels.max()  # truth label ranges
                bins = np.linspace(lo, hi, num_bins + 1)
                bin_width = (bins[1] - bins[0]) / 2
                bin_centers = (bins[1:] + bins[:-1]) / 2
                bin_idcs = np.digitize(labels, bins)
                partitions = [preds[bin_idcs == i + 1] for i in range(num_bins)]
                mares = abs(preds - labels) / labels
                mare_partitions = [mares[bin_idcs == i + 1] for i in range(num_bins)]
                MARE = mares.mean()

                if self.cfg.gaussian:
                    errs = [
                        sigmas[bin_idcs == i + 1] * (hi - lo) for i in range(num_bins)
                    ]
                    errs = list(map(np.mean, errs))
                else:
                    errs = list(map(np.std, partitions))

                # fill main axis
                pad = 0.04 * (hi - lo)
                main_ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], **ref_kwargs)
                main_ax.errorbar(
                    bin_centers, list(map(np.mean, partitions)), yerr=errs, **err_kwargs
                )
                main_ax.text(0.1, 0.9, f"{MARE=:.1e}", transform=main_ax.transAxes)

                # fill ratio axis
                ratio_ax.errorbar(
                    bin_centers,
                    list(map(np.mean, mare_partitions)),
                    yerr=list(map(np.std, mare_partitions)),
                    **err_kwargs,
                )
                ratio_ax.axhline(y=MARE, color="navy", ls="--", lw=1)
                ratio_ax.semilogy()

                # axis labels
                param_idx = self.cfg.target_indices[i]
                main_ax.set_title(PARAM_NAMES[param_idx], fontsize=14)
                main_ax.set_ylabel("Network", fontsize=13)
                ratio_ax.set_ylabel(
                    r"$\left|\frac{\text{Net}\,-\,\text{True}}{\text{True}}\right|$",
                    fontsize=10,
                )
                ratio_ax.set_xlabel("Truth", fontsize=13)

                # axis limits
                main_ax.set_xlim([lo - pad, hi + pad])
                main_ax.set_ylim([lo - pad, hi + pad])
                ratio_ax.set_xlim(*main_ax.get_xlim())
                # ratio_ax.set_ylim(1e-3, 1)

                # clean
                main_ax.set_xticklabels([])
                fig.tight_layout()

                pdf.savefig(fig, bbox_inches="tight")

        self.log.info(f"Saved plots to {savepath}")