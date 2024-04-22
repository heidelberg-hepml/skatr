import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

from src.utils import ensure_device
from .. import transforms

log = logging.getLogger('Trainer')

class Trainer:

    def __init__(
            self,
            model:nn.Module,
            dataloaders:Dict[str, DataLoader],
            cfg:DictConfig,
            exp_dir:str,
            device:str
        ):
        """
        model       -- a pytorch model to be trained
        dataloaders -- a dictionary containing pytorch data loaders at keys 'train' and 'val'
        cfg         -- configuration dictionary
        exp_dir     -- directory to which training outputs will be saved
        """

        self.model = model
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.device = device
        self.exp_dir = exp_dir
        self.transformations = [
            getattr(transforms, name)(**kwargs) for name, kwargs in self.cfg.preprocessing.items()
        ]    

    def prepare_training(self):
        
        log.info('Preparing model training')

        # TODO: Allow for selecting optimizer in config
        self.optimizer = torch.optim.AdamW(
            self.model.net.parameters(), lr=self.cfg.lr, betas=self.cfg.adam.betas,
            weight_decay=self.cfg.adam.weight_decay,
        )

        self.steps_per_epoch = len(self.dataloaders['train'])
        if self.cfg.use_scheduler:
            raise NotImplementedError
            self.scheduler = set_scheduler(
                self.optimizer, self.cfg, steps_per_epoch=self.steps_per_epoch
            )

        if self.cfg.use_tensorboard:
            self.summarizer = SummaryWriter(self.exp_dir)
            log.info(f'Writing tensorboard summaries to dir {self.exp_dir}')
        else:
            log.info('`use_tensorboard` set to False. No summaries will be written')

        self.epoch_train_losses = np.array([])
        self.epoch_val_losses = np.array([])

    def run_training(self):

        self.prepare_training()

        epochs = self.cfg.epochs
        if start_epoch := self.cfg.start_epoch:
            self.load(epoch=start_epoch) # TODO: Load model from checkpoint
            # self.scheduler = set_scheduler(
            #     self.optimizer, self.params, self.steps_per_epoch,
            #     last_epoch=start_epoch*self.n_trainbatches
            # )
            log.info(f'Warm starting training from epoch {start_epoch}')
        
        log.info(f'Beginning training loop with epochs set to {epochs}')
        t_0 = time.time()
        for e in range(epochs):
            
            t0 = time.time()
            self.epoch = (start_epoch or 0) + e

            # train
            self.model.net.train()
            self.train_one_epoch()

            # validate at given frequency
            if (self.epoch + 1) % self.cfg.validate_freq == 0:
                self.model.eval()
                self.validate_one_epoch()

            # optionally save model at given frequency
            if save_freq := self.cfg.save_freq:
                if (self.epoch + 1) % save_freq == 0 or self.epoch == 0:
                    self.save(epoch=f"{self.epoch}")

            # estimate training time
            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * epochs
                log.info(f'Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h')
            
        t_1 = time.time()
        traintime = t_1 - t_0
        log.info(
            f'Finished training {epochs} epochs after {traintime:.2f} s'
            f' = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h.'
        )
        
        # save final model
        log.info('Saving final model')
        self.model.eval()
        self.save()

    def train_one_epoch(self):
        
        # create list to save loss per iteration
        train_losses = []
        
        # iterate batch wise over input
        for itr, x in enumerate(self.dataloaders['train']):

            # clear optimizer gradients
            self.optimizer.zero_grad(set_to_none=True)

            # place x on device
            x = ensure_device(x, self.device)

            # preprocess
            for transform in self.transformations:
                x = transform.forward(*x)

            # calculate batch loss
            loss = self.model.batch_loss(x)
            # check for nans / infs in loss
            loss_numpy = loss.detach().cpu().numpy()
            if ~np.isfinite(loss_numpy):
                log.info(f"Unstable loss. Skipping backprop for epoch {self.epoch}")
                continue

            # propagate gradients
            loss.backward()
            # optionally clip gradients
            if clip := self.cfg.gradient_norm:
                nn.utils.clip_grad_norm_(self.model.net.parameters(), clip)
            # update weights
            self.optimizer.step()
            
            # update learning rate
            if self.cfg.use_scheduler:
                self.scheduler.step()
            
            # track loss
            train_losses.append(loss_numpy)
            if self.cfg.use_tensorboard:
                self.summarizer.add_scalar(
                    "iter_loss_train", train_losses[-1], itr + self.epoch*self.steps_per_epoch
                )
        
        # track loss
        self.epoch_train_losses = np.append(self.epoch_train_losses, np.mean(train_losses))

        # optionally log to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar("epoch_loss_train", self.epoch_train_losses[-1], self.epoch)
            if self.cfg.use_scheduler:
                self.summarizer.add_scalar(
                    "learning_rate", self.scheduler.get_last_lr()[0],self.epoch
                )

    @torch.inference_mode()
    def validate_one_epoch(self):
        
        # calculate loss batchwise over input
        val_losses = []
        for x in self.dataloaders['val']:

            # place x on device
            x = ensure_device(x, self.device)
            
            # preprocess
            for transform in self.transformations:
                x = transform.forward(*x)

            loss = self.model.batch_loss(x).detach().cpu().numpy()
            val_losses.append(loss)

        # track loss
        self.epoch_val_losses = np.append(self.epoch_val_losses, np.mean(val_losses))
       
       # optional logging to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar("epoch_loss_val", self.epoch_val_losses[-1], self.epoch)

    def save(self, epoch=''):
        """Save the model along with the training state"""
        state_dicts = {
            'opt': self.optimizer.state_dict(),
            'net': self.model.net.state_dict(),
            'losses': self.epoch_train_losses,
            'epoch': self.epoch
        }
        if self.cfg.use_scheduler:
            state_dicts['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dicts, os.path.join(self.exp_dir, f'model{epoch}.pt'))

    def load(self, epoch=''):
        """Load the model and training state"""
        name = os.path.join(self.exp_dir, f'model{epoch}.pt')
        state_dicts = torch.load(name, map_location=self.device)
        self.model.net.load_state_dict(state_dicts['net'])
        
        if 'losses' in state_dicts:
            self.epoch_train_losses = state_dicts.get('losses', {})
        if 'epoch' in state_dicts:
            self.epoch = state_dicts.get('epoch', 0)
        if 'opt' in state_dicts:
           self.optimizer.load_state_dict(state_dicts['opt'])
        if 'scheduler' in state_dicts:
           self.scheduler.load_state_dict(state_dicts['scheduler'])
        self.model.net.to(self.device)