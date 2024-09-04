import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import Dataset

from src.experiments.base_experiment import BaseExperiment
from src.models import Classifier
from src.utils.plotting import PARAM_NAMES

class ClassificationExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return ClassificationDatasetByFile(self.cfg.data)
        else:
            return ClassificationDataset(self.cfg.data, self.device)

    def get_model(self):
        return Classifier(self.cfg)
    
    def plot(self):
        label_pred_pairs = np.load(os.path.join(self.exp_dir, 'label_pred_pairs.npy'))
        
        # check for existing plots
        savename = 'auc.pdf'
        savepath = os.path.join(self.exp_dir, savename)
        if os.path.exists(savepath):
            old_dir = os.path.join(self.exp_dir, 'old_plots')
            self.log.info(f'Moving old plots to {old_dir}')
            os.makedirs(old_dir, exist_ok=True)
            os.rename(savepath, os.path.join(old_dir, savename))

        # # create plots
        # with PdfPages(savepath) as pdf:

        #     # iterate over individual parameters
            
        #     for i in range(label_pred_pairs.shape[1]):

        #         fig, ax = plt.subplots(figsize=(5,4))
        #         labels, preds = label_pred_pairs[:, i].T

        #         lo, hi = labels.min(), labels.max() # range of true targets
        #         NRMSE = np.sqrt(((labels-preds)**2).mean())/(hi-lo)
                
        #         pad = 0.1*(hi-lo)
        #         ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color='crimson', ls='--', lw=2)
        #         ax.scatter(labels, preds, alpha=0.4, color='darkblue')
        #         ax.text(0.1, 0.9, f"{NRMSE=:.3f}", transform=ax.transAxes)

        #         ax.set_xlabel(PARAM_NAMES[i])
        #         ax.set_ylabel(f'Prediction')
        #         fig.tight_layout()

        #         pdf.savefig(fig)

        self.log.info(f'Saved plots to {savepath}')
    
    @torch.inference_mode()
    def evaluate(self, dataloaders):
        """
        Evaluates the Classifier on lightcones in the test dataset.
        Predictions are saved alongside truth labels
        """
        
        # disable batchnorm updates, dropout etc.
        self.model.eval()

        # get truth targets and predictions across the test set
        labels, preds = [], []
        for x, y in dataloaders['test']:

            labels.append(y.numpy())
            

            # predict
            pred = self.model.predict(x).detach().cpu()
            
            # append prediction
            preds.append(pred.numpy())

        # stack results
        labels = np.vstack(labels)
        # preprocess input
        for transform in self.preprocessing['y']:
            labels = transform.reverse(labels)

        preds = np.vstack(preds)
        
        # save results
        savepath = os.path.join(self.exp_dir, 'label_pred_pairs.npy')
        self.log.info(f'Saving label/prediction pairs to {savepath}')
        np.save(savepath, np.stack([labels, preds], axis=-1))


class ClassificationDatasetByFile(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        self.files0 = sorted(glob(f'{cfg.dir0}/run*.npz'))
        self.files1 = sorted(glob(f'{cfg.dir1}/run*.npz'))
        if len(self.files0) > (lim := len(self.files1)):
            self.files0 = self.files0[:lim]
        if len(self.files1) > (lim := len(self.files0)):
            self.files1 = self.files1[:lim]            
        self.files = self.files0 + self.files1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        record = np.load(self.files[idx])
        label = int(idx>=len(self.files0))
        X = torch.from_numpy(record['image']).to(torch.get_default_dtype())
        y = torch.tensor([label]).to(torch.get_default_dtype())
        return X, y