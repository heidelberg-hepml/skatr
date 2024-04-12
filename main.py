#!/usr/bin/env python3
import hydra
import logging
import sys
from omegaconf import DictConfig

from src import experiments
from src import utils

log = logging.getLogger(__name__)

@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg:DictConfig):
    
    if cfg.submit:
        # submit this script to a cluster
        if cfg.cluster.scheduler == 'pbs':
            exec_cmd = utils.submit_pbs(cfg)
            log.info(f'Executing in shell: {exec_cmd}')

    else:
        # select experiment
        exp_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        try:
            exp_cls = getattr(experiments, cfg.experiment)
        except AttributeError:
            log.error(f'Experiment "{cfg.experiment}" not recognized!')
            sys.exit()

        # run experiment
        experiment = exp_cls(cfg, exp_dir)
        experiment.run()

if __name__ == '__main__':
    main()
