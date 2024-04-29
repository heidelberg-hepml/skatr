#!/usr/bin/env python3

import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src import experiments
from src import utils

log = logging.getLogger(__name__)

@hydra.main(config_path='config', config_name='regression_mini', version_base=None)
def main(cfg:DictConfig):
    
    hcfg = HydraConfig.get()
    exp_dir = hcfg.runtime.output_dir if cfg.training else cfg.prev_exp_dir
    if cfg.submit:
        # submit this script to a cluster
        if cfg.cluster.scheduler == 'pbs':
            exec_cmd = utils.submit_pbs(cfg, hcfg, exp_dir)
            log.info(f'Executing in shell: {exec_cmd}')
    else:
        # select experiment
        exp_cls = getattr(experiments, cfg.experiment)
        experiment = exp_cls(cfg, exp_dir)

        # run experiment
        experiment.run()

if __name__ == '__main__':
    main()
