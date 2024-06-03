import logging
import torch
from abc import abstractmethod
from torch.utils.data import DataLoader, random_split

from .. import transforms
from ..trainers.trainer import Trainer

log = logging.getLogger('Experiment')

class BaseExperiment:

    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.device = f'cuda:{cfg.device}' if cfg.use_gpu else 'cpu'
        self.exp_dir = exp_dir
        torch.set_default_dtype(getattr(torch, cfg.dtype))

        self.preprocessing = [
            getattr(transforms, name)(**kwargs) for name, kwargs in self.cfg.preprocessing.items()
        ]

    def run(self):

        log.info('Reading data')
        dataset = self.get_dataset() # TODO: Print the dataset signature/shape

        log.info('Initializing dataloaders')
        dataloaders = self.get_dataloaders(dataset)

        log.info(f'Using device {self.device}')
        log.info('Initializing model')
        model = self.get_model().to(device=self.device)
        # TODO: Implement option for memory format in trainer
        log.info(
            f'Model ({model.__class__.__name__}[{model.net.__class__.__name__}]) has '
            f'{sum(w.numel() for w in model.trainable_parameters)} trainable parameters'
        )

        if self.cfg.train:
            log.info('Initializing trainer')
            trainer = Trainer(
                model, dataloaders, self.preprocessing, self.cfg.training, self.exp_dir, self.device
            )
            log.info('Running training')
            trainer.run_training()
        else:
            log.info(f'Loading model state from {self.cfg.prev_exp_dir}.')
            model.load(self.exp_dir, self.device)
            model.eval()

        if self.cfg.evaluate:
            log.info('Running evaluation')
            self.evaluate(dataloaders, model)

        if self.cfg.plot:
            log.info('Making plots')
            self.plot()
    
    def get_dataloaders(self, dataset):

        assert sum(self.cfg.data.splits.values()) == 1.
        
        # partition the dataset (using a seed to fix the test split)
        trn = self.cfg.data.splits.train
        val = self.cfg.data.splits.val
        tst = self.cfg.data.splits.test
        trainval_set, test_set = random_split(
            dataset, [trn + val, tst], generator=torch.Generator().manual_seed(1729)
        )
        train_set, val_set = random_split(trainval_set, [trn/(trn+val), val/(trn+val)])
        dataset_splits = {'train': train_set, 'val': val_set, 'test': test_set}
        del dataset, trainval_set # TODO: Assess if this is really necessary

        # create dataloaders
        dataloaders = {
            k: DataLoader(
                d, batch_size=self.cfg.training.batch_size, shuffle=True, drop_last=True,
                num_workers=self.cfg.num_cpus, pin_memory=False # pinning can cause memory issues
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