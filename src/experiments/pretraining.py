import torch

from src import models
from src.experiments.base_experiment import BaseExperiment
from src.utils import datasets

class PretrainingExperiment(BaseExperiment):
    
    def get_dataset(self, directory):
        prep = self.preprocessing
        if self.cfg.data.file_by_file:
            return datasets.LCDatasetByFile(
                self.cfg.data, directory, preprocessing=prep, use_labels=False
            )
        else:
            return datasets.LCDataset(
                self.cfg.data, directory, self.device, preprocessing=prep, use_labels=False
            )

    def get_model(self):
        model_cls = getattr(models, self.cfg.model)
        return model_cls(self.cfg)
    
    def plot(self):
        raise NotImplementedError
    
    @torch.inference_mode()
    def evaluate(self, dataloaders, model):
        """Use the pretrained summary to compress the chosen dataset"""

        # free memory from current dataset
        # load specific dataset as LabelledDataset
        # iterate in batches and compress with summary
        # stack results over batches (and data splits?)
        # save with labels in hdf5 format
        raise NotImplementedError