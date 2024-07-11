import os
from omegaconf import OmegaConf, open_dict

def get_prev_config(prev_exp_dir):
    return OmegaConf.load(os.path.join(prev_exp_dir, '.hydra/config.yaml'))

def update_config_from_prev(cfg, hydra_cfg, prev_exp_dir):
    prev_cfg = get_prev_config(prev_exp_dir)
    overrides = OmegaConf.from_dotlist(OmegaConf.to_object(hydra_cfg.overrides.task))
    with open_dict(cfg):
        cfg = OmegaConf.merge(cfg, prev_cfg, overrides)
    return cfg