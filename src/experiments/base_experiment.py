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

        # warnings:
        if cfg.data.summarize and not self.cfg.frozen_backbone:
            self.log.warn(f'Asking to summarize dataset, but backbone is not frozen')

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
            
            self.log.info('Initializing dataloaders')
            dataloaders = self.get_dataloaders()#dataset, dataset_test=dataset_test)

        if self.cfg.train:
            
            tcfg = self.cfg.training

            augs = self.get_augmentations()
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
    
    def get_dataloaders(self):
        
        dcfg = self.cfg.data
        fixed_rng = torch.Generator().manual_seed(1729)
        
        # read data
        dataset = self.get_dataset(dcfg.dir)
        dataset_test = (self.get_dataset(dcfg.dir + '/test') if dcfg.use_test_dir else None)

        shape_strings = [
            f'({len(dataset)}, {repr(tuple(x.shape))[1:].replace(',)', ')')}'
            for x in dataset[0]
        ]
        self.log.info(f"Read data with shapes {', '.join(shape_strings)}")
        
        # partition the dataset into train/val/test
        trn = dcfg.splits.train
        if dataset_test is None:
            tst = dcfg.splits.test
            val = 1 - trn - tst
            assert val > 0, 'A validation split is required'

            dataset_splits = dict(zip(
                ('train', 'val', 'test'),
                random_split(dataset, [trn, val, tst], generator=fixed_rng)
            ))
        else:
            val = 1 - trn 
            assert val > 0, 'A validation split is required'
            
            dataset_train, dataset_val = random_split(dataset, [trn, val], generator=fixed_rng)
            dataset_splits = {
                'train': dataset_train, 'val': dataset_val, 'test': dataset_test
            }

        del dataset #TODO: Assess if this is really necessary
        
        # create dataloaders
        dataloaders = {}
        num_cpus = 0 if dcfg.on_gpu or dcfg.summarize else self.cfg.num_cpus
        self.log.info(f'{num_cpus=}')
        for k, d in dataset_splits.items():
            
            # optionally summarize (compress) dataset
            if self.cfg.backbone and dcfg.summarize:
                dataset_splits[k] = SummarizedLCDataset(
                    d, summary_net=self.model.bb, device=self.device,
                    summary_batch_size=dcfg.summary_batch_size, num_cpus=self.cfg.num_cpus,
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
    
    def get_augmentations(self):
        augs = []
        if self.cfg.training.augment and not self.cfg.data.summarize:
            for name, kwargs in self.cfg.training.augmentations.items():
                aug = getattr(augmentations, name)(**kwargs)
                augs.append(aug)
        return augs
    
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