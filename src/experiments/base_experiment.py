import logging
import torch
from abc import abstractmethod
from torch.utils.data import DataLoader, random_split

from ..trainers.trainer import Trainer

log = logging.getLogger('Experiment')

class BaseExperiment:

    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.device = f'cuda:{cfg.device}' if cfg.use_gpu else 'cpu'
        self.exp_dir = exp_dir

        torch.set_default_dtype(getattr(torch, cfg.dtype))

    def run(self):

        log.info('Reading data and creating dataloaders')
        # TODO: Add preprocessing
        # if cfg.transforms:
        #     self.transforms = get_transformations(self.cfg.transforms, exp_dir=exp_dir)
        dataset = self.get_dataset()
        dataloaders = self.get_dataloaders(dataset)
        log.info(f'Dataset has signature {dataset.__annotations__}')

        log.info('Initializing model')
        model = self.get_model().to(device=self.device, memory_format=torch.channels_last_3d)
        log.info(
            f'Model ({model.__class__.__name__}[{model.net.__class__.__name__}]) has '
            f'{sum(w.numel() for w in model.parameters())} parameters'
        )

        if self.cfg.train:
            log.info('Initializing trainer')
            trainer = Trainer(model, dataloaders, self.cfg.training, self.exp_dir, self.device)
            log.info('Running training')
            trainer.run_training()

        if self.cfg.evaluate:
            log.info('Running evaluation')
            self.evaluate(dataloaders)

        if self.cfg.plot:
            log.info('Making plots')
            self.plot()
    
    def get_dataloaders(self, dataset):

        assert sum(self.cfg.data.splits.values()) == 1.
        
        # determine sizes of each dataset split
        data_size = len(dataset)
        split_sizes = {
            'train': int(data_size*self.cfg.data.splits.train),
            'val'  : int(data_size*self.cfg.data.splits.val),
            'test' : int(data_size*self.cfg.data.splits.test),
        }
        
        # TODO: Update this to move to separated train/test files

        # split the dataset
        dataset_splits = dict(zip(split_sizes, random_split(dataset, split_sizes.values())))
        del dataset

        for k, v in split_sizes.items():
            if (rem := v % self.cfg.training.batch_size):
                log.warn(f'\'{k}\' dataloader will drop the last batch (with {rem} data points)')

        # create dataloaders
        dataloaders = {
            k: DataLoader(
                d, batch_size=self.cfg.training.batch_size, shuffle=True, drop_last=True,
                pin_memory=not self.cfg.data.on_gpu
            ) for k, d in dataset_splits.items()
        }

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

        

