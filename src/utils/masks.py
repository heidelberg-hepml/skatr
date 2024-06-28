import math
import torch

def random_patch_mask(num_patches, cfg, batch_size, device):
    """
    :param num_patches: iterable containing the number of patches per dimension
    
    Returns a boolean mask with shape (batch_size, prod(num_patches)), where the proportion of
    `True` entries is `cfg.mask_frac`.
    """
    # get total number of patches
    T = math.prod(num_patches)
    # get number of patches to be masked
    num_masked = int(cfg.mask_frac * T)
    # uniformly sample `num_masked` integers per batch
    mask_idcs = torch.rand(batch_size, T, device=device).topk(k=num_masked, dim=-1).indices
    # construct boolean mask
    mask = torch.zeros((batch_size, T), device=device).scatter_(-1, mask_idcs, 1).bool()
    return mask