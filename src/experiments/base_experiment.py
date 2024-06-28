import logging
import torch
from abc import abstractmethod
from torch.utils.data import DataLoader, random_split

from src.utils import transforms
from src.utils.trainer import Trainer

class BaseExperiment:

    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.device = f'cuda:{cfg.device}' if cfg.use_gpu else 'cpu'
        self.exp_dir = exp_dir
        torch.set_default_dtype(getattr(torch, cfg.dtype))

        self.preprocessing={ # initialize preprocessing transforms for data and targets
            k: [getattr(transforms, name)(**kwargs) for name, kwargs in d.items()]
            for k, d in self.cfg.preprocessing.items()
        }

        self.log = logging.getLogger('Experiment')

    def run(self):

        self.log.info('Reading data')
        dataset = self.get_dataset() # TODO: Print the dataset signature/shape

        self.log.info('Initializing dataloaders')
        dataloaders = self.get_dataloaders(dataset)

        self.log.info(f'Using device {self.device}')
        self.log.info('Initializing model')
        if self.cfg.train or self.cfg.evaluate:
            model = self.get_model().to(device=self.device)
            # TODO: Implement option for memory format in trainer
            self.log.info(
                f'Model ({model.__class__.__name__}[{model.net.__class__.__name__}]) has '
                f'{sum(w.numel() for w in model.trainable_parameters)} trainable parameters'
            )

        if self.cfg.train:
            self.log.info('Initializing trainer')
            trainer = Trainer(
                model, dataloaders, self.preprocessing, self.cfg.training, self.exp_dir, self.device
            )
            self.log.info('Running training')
            trainer.run_training()
        elif self.cfg.evaluate:
            self.log.info(f'Loading model state from {self.cfg.prev_exp_dir}.')
            model.load(self.exp_dir, self.device)
            model.eval()

        if self.cfg.evaluate:
            self.log.info('Running evaluation')
            self.evaluate(dataloaders, model)

        if self.cfg.plot:
            self.log.info('Making plots')
            self.plot()
    
    def get_dataloaders(self, dataset):

        assert sum(self.cfg.data.splits.values()) == 1.
        
        # partition the dataset (using a seed to fix the test split)
        trn = self.cfg.data.splits.train
        val = self.cfg.data.splits.val
        tst = self.cfg.data.splits.test
        dataset_splits = dict(zip(
            ('train', 'val', 'test'), random_split(
                dataset, [trn, val, tst], generator=torch.Generator().manual_seed(1729)
            )
        ))
        # trainval_set, test_set = random_split(
        #     dataset, [trn + val, tst], generator=torch.Generator().manual_seed(1729)
        # )
        # train_set, val_set = random_split(trainval_set, [trn/(trn+val), val/(trn+val)])        
        # dataset_splits = {'train': train_set, 'val': val_set, 'test': test_set}
        del dataset#, trainval_set # TODO: Assess if this is really necessary

        # create dataloaders
        dataloaders = {
            k: DataLoader(
                d, shuffle=k=='train', drop_last=True, num_workers=self.cfg.num_cpus,
                pin_memory=False, # pinning can cause memory issues with large lightcones
                batch_size=(
                    self.cfg.training.batch_size if k=='train'
                    else self.cfg.training.test_batch_size
                )
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