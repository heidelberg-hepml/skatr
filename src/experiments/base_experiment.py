import logging
import torch
from abc import abstractmethod
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split, Subset

from src.utils.datasets import SummarizedLCDataset
from src.utils.trainer import Trainer

class BaseExperiment:

    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.device = f'cuda:{cfg.device}' if cfg.use_gpu else 'cpu'
        self.exp_dir = exp_dir
        self.log = logging.getLogger('Experiment')
        torch.set_default_dtype(getattr(torch, cfg.dtype))

        self.preprocessing={ # initialize preprocessing transforms for data and targets
            k: [instantiate(t) for t in ts] for k, ts in self.cfg.preprocessing.items()
        }
        transform_names = {
            k: [t.__class__.__name__ for t in ts] for k,ts in self.preprocessing.items()
        }
        self.log.info(f'Loaded preprocessing dict: {transform_names}')

        if cfg.data.sequential_splits: # partition splits either sequentially or randomly
            self.split_func = lambda dataset, split_sizes: self.sequential_split(dataset, split_sizes)
        else:
            self.split_func = lambda dataset, split_sizes: random_split(dataset, split_sizes, generator=torch.Generator().manual_seed(1729))

    def run(self):

        self.log.info(f'Using device {self.device}')
        self.log.info('Initializing model')
        if self.cfg.train or self.cfg.evaluate:
            model = self.get_model().to(device=self.device)
            # TODO: Implement option for memory format in trainer
            self.log.info(
                f'Model ({model.__class__.__name__}[{model.net.__class__.__name__}]) has '
                f'{sum(w.numel() for w in model.trainable_parameters)} trainable parameters'
            )

        self.log.info('Reading and preprocessing data')
        dataset = self.get_dataset() # TODO: Print the dataset signature/shape

        # optionally summarize (compress) dataset
        if self.cfg.backbone and self.cfg.frozen_backbone: # TODO: embed in batches ... how?
            self.log.info('Summarizing dataset')

            dataset = SummarizedLCDataset(
                dataset, summary_net=model.bb, device=self.device, augment=self.cfg.training.augment
            )

        self.log.info('Initializing dataloaders')
        dataloaders = self.get_dataloaders(dataset)

        if self.cfg.train:
            self.log.info('Initializing trainer')
            trainer = Trainer(
                model, dataloaders, self.preprocessing, self.cfg.training, self.exp_dir, self.device
            )
            self.log.info('Running training')
            trainer.run_training()

        if self.cfg.evaluate:
            self.log.info(f'Loading model state from {self.exp_dir}.')
            model.load(self.exp_dir, self.device)
            model.eval()
            self.log.info('Running evaluation')
            self.evaluate(dataloaders, model)

        if self.cfg.plot:
            self.log.info('Making plots')
            self.plot()
    
    def get_dataloaders(self, dataset):

        assert sum(self.cfg.data.splits.values()) == 1.
        
        # partition the dataset using self.split_func
        trn = self.cfg.data.splits.train
        val = self.cfg.data.splits.val
        tst = self.cfg.data.splits.test
        dataset_splits = dict(zip(
            ('train', 'val', 'test'), self.split_func(dataset, split_sizes=[trn, val, tst])
        ))
        split_sizes = [trn, val, tst]

        # trainval_set, test_set = random_split(
        #     dataset, [trn + val, tst], generator=torch.Generator().manual_seed(1729)
        # )
        # train_set, val_set = random_split(trainval_set, [trn/(trn+val), val/(trn+val)])        
        # dataset_splits = {'train': train_set, 'val': val_set, 'test': test_set}
        del dataset#, trainval_set # TODO: Assess if this is really necessary

        # create dataloaders
        num_cpus = 0 if self.cfg.data.on_gpu or self.cfg.frozen_backbone else self.cfg.num_cpus
        dataloaders = {
            k: DataLoader(
                d, shuffle=k=='train', drop_last=True, pin_memory=False, # pinning can cause memory issues with large lightcones
                num_workers=num_cpus, # parallel loading from GPU causes CUDA error
                batch_size=(
                    self.cfg.training.batch_size if k=='train'
                    else self.cfg.training.test_batch_size
                )
            ) for k, d in dataset_splits.items()
        }

        return dataloaders

    def sequential_split(self, dataset, split_sizes):
        dataset_size = len(dataset.files)
        splits = []
        start = 0
        for size in split_sizes:
            stop = start + round(size * dataset_size)
            splits.append(Subset(dataset, range(start, stop)))
            start = stop
        return splits
    
    @abstractmethod
    def get_dataset(self):
        pass
    
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def plot(self):
        pass
    
    @abstractmethod
    def evaluate(self, dataloaders):
        pass