import torch

from src import models
from src.experiments.base_experiment import BaseExperiment
from src.utils import datasets

class PretrainingExperiment(BaseExperiment):
    
    def get_dataset(self):
        if self.cfg.data.file_by_file:
            return datasets.UnlabelledDatasetByFile(self.cfg.data)
        else:
            return datasets.UnlabelledDataset(self.cfg.data, self.device)

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