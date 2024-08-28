import logging
import torch
from abc import abstractmethod
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split, Subset

from src.utils import augmentations
from src.utils.datasets import SummarizedLCDataset
from src.utils.trainer import Trainer

class BaseExperiment:

    def __init__(self, cfg, exp_dir):
        
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.log = logging.getLogger('Experiment')
        torch.set_default_dtype(getattr(torch, cfg.dtype))

        self.device = (
            f'cuda:{cfg.device}' if cfg.use_gpu and torch.cuda.is_available()
            else f'mps:{cfg.device}' if cfg.use_gpu else 'cpu'
        )
        self.log.info(f'Using device {self.device}')

        self.preprocessing={ # initialize preprocessing transforms for data and targets
            k: [instantiate(t) for t in ts] for k, ts in self.cfg.preprocessing.items()
        }
        transform_names = {
            k: [t.__class__.__name__ for t in ts] for k,ts in self.preprocessing.items()
        }
        self.log.info(f'Loaded preprocessing dict: {transform_names}')

    def run(self):
        
        self.log.info('Initializing model')
        if self.cfg.train or self.cfg.evaluate:
            
            model = self.get_model().to(device=self.device)
            self.model = model
            self.log.info(
                f'Model ({model.__class__.__name__}[{model.net.__class__.__name__}]) has '
                f'{sum(w.numel() for w in model.trainable_parameters)} trainable parameters'
            )

        if self.cfg.train or self.cfg.evaluate:
            
            self.log.info('Reading and preprocessing data')
            dataset = self.get_dataset(self.cfg.data.dir) # TODO: Print the dataset signature/shape

            dataset_test = (
                self.get_dataset(self.cfg.data.dir + '/test')
                if self.cfg.data.use_test_dir else None
            )

            self.log.info('Initializing dataloaders')
            dataloaders = self.get_dataloaders(dataset, dataset_test=dataset_test)

        if self.cfg.train:
            
            tcfg = self.cfg.training

            # init augmentations
            augs = []
            if tcfg.augment and not (self.cfg.backbone and self.cfg.frozen_backbone):
                for name, kwargs in tcfg.augmentations.items():
                    aug = getattr(augmentations, name)(**kwargs)
                    augs.append(aug)
                
                self.log.info(
                    f"Loaded augmentations: {', '.join([a.__class__.__name__ for a in augs])}"
                )

            self.log.info('Initializing trainer')                
            trainer = Trainer(
                model, dataloaders, self.preprocessing, augs, tcfg, self.exp_dir, self.device
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
    
    def get_dataloaders(self, dataset, dataset_test=None):
        
        fixed_rng = torch.Generator().manual_seed(1729)

        if dataset_test is None:
            # partition the dataset using self.split_func
            trn = self.cfg.data.splits.train
            tst = self.cfg.data.splits.test
            val = 1 - trn - tst
            assert val > 0, 'A validation split is required'

            dataset_splits = dict(zip(
                ('train', 'val', 'test'),
                random_split(dataset, [trn, val, tst], generator=fixed_rng)
            ))
        else:
            trn = self.cfg.data.splits.train
            val = 1 - trn 
            assert val > 0, 'A validation split is required'
            
            dataset_train, dataset_val = random_split(dataset, [trn, val], generator=fixed_rng)
            dataset_splits = {
                'train': dataset_train, 'val': dataset_val, 'test': dataset_test
            }

        del dataset #TODO: Assess if this is really necessary
        
        # create dataloaders
        dataloaders = {}
        num_cpus = 0 if self.cfg.data.on_gpu or self.cfg.frozen_backbone else self.cfg.num_cpus
        for k, d in dataset_splits.items():
            
            # optionally summarize (compress) dataset
            if self.cfg.backbone and self.cfg.frozen_backbone:
                dataset_splits[k] = SummarizedLCDataset(
                    d, summary_net=self.model.bb, device=self.device,
                    augment=self.cfg.training.augment and k=='train' # only augment training split
                )

            batch_size = (self.cfg.training.batch_size if k=='train'
                          else self.cfg.training.test_batch_size)

            dataloaders[k] = DataLoader(
                dataset_splits[k], shuffle=k=='train', drop_last=k=='train', batch_size=batch_size,
                pin_memory=False, # pinning can cause memory issues with large lightcones
                num_workers=num_cpus # parallel loading from GPU causes CUDA error
            )

        return dataloaders
    
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