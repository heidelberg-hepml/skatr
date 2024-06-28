import math
import torch

def random_patch_mask(num_patches, cfg, batch_size, device):
    """
    :param num_patches: iterable containing the number of patches per dimension
    
    Returns a Masks x by randomly selecting patches in each batch and replacing their
    embedding with `self.mask_token`. The number of patches to mask is 
    determined by the `cfg.mask_frac` option.
    """

    # get total number of patches
    T = math.prod(num_patches)
    # get number of patches to be masked
    num_masked = int(cfg.mask_frac * T)
    # uniformly sample `num_masked` integers per batch
    mask_idcs = torch.rand(batch_size, T, device=device).topk(k=num_masked, dim=-1).indices
    return mask_idcs