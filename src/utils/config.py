import os
import sys
from omegaconf import OmegaConf, open_dict

def get_prev_config(prev_exp_dir):
    return OmegaConf.load(os.path.join(prev_exp_dir, '.hydra/config.yaml'))

def update_config_from_prev(cfg, hydra_cfg, prev_exp_dir):
    prev_cfg = get_prev_config(prev_exp_dir)
    overrides = OmegaConf.from_dotlist(OmegaConf.to_object(hydra_cfg.overrides.task))
    # TODO: Using cfg could let outdated runs be loaded without having to edit their cfg.
    # with open_dict(cfg):
        # cfg = OmegaConf.merge(cfg, prev_cfg, overrides)
    # return cfg
    return OmegaConf.merge(prev_cfg, overrides)

def check_cfg(cfg, log):

    # warn
    if cfg.data.summarize and not cfg.summary_net.frozen:
        log.warn(f'Summary net is not frozen, but asking to summarize dataset.')
    
    if cfg.num_cpus and not cfg.data.file_by_file:
        log.warn(f'Using {cfg.num_cpus} cpus not reading from disk. Training may be slower.')

    # exit
    if (cfg.prev_exp_dir and cfg.train) and not cfg.training.warm_start:
        log.error(
            'Rerunning experiment with train=True but warm_start=False. Exiting to avoid overwrite.'
        )
        sys.exit()