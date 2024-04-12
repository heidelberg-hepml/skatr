import torch.nn as nn
import logging
from omegaconf import DictConfig

from .. import networks

log = logging.getLogger(__name__)

class Model(nn.Module):

    def __init__(self, cfg:DictConfig):
        super().__init__()
        try:
            self.net = getattr(networks, cfg.net.arch)(cfg.net)
        except AttributeError as e:
            log.error(f'Network architecture "{cfg.net.arch}" not recognized!')
            raise e

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
