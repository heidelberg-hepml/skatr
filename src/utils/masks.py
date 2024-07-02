import math
import torch

def patch_mask(num_patches, cfg, batch_size, device):
    """
    :param num_patches: iterable containing the number of patches per dimension

    Returns a tensor with shape (B [batch size], K ]) containing random indices in the range [0,T)
    with T = prod(num_patches) and K = T * cfg.mask_frac.
    """
    # get total number of patches
    T = math.prod(num_patches)
    # get number of patches to be masked
    num_masked = int(cfg.mask_frac * T)
    # uniformly sample `num_masked` integers per batch
    mask_idcs = torch.rand(batch_size, T, device=device).topk(k=num_masked, dim=-1).indices
    return mask_idcs

def multiblock_mask(num_patches, cfg, batch_size, device):
    """
    :cfg.num_targets: Number of targets to predict

    Returns indices of target masks and context masks. Targets are either short range or long range 
    with different block sampling parameters respectively. The contexts are the complements of their targets.
    """

    # sample block size (once per batch)
    block_sizes = [
        sample_block_size(num_patches, cfg, mode='short' if i%2 else 'long')
        for i in range(cfg.num_targets)
    ]

    batch = []
    for _ in range(batch_size):
        target_masks = []
        context_masks = []
        for i, block_size in enumerate(block_sizes):
            if i%2:
                mode = 'short'
                num_blocks = cfg.short_num_blocks
            else:
                mode = 'long'
                num_blocks = cfg.long_num_blocks

            target_mask, context_mask = block_mask(num_patches, cfg, block_size, mode, device)
            for _ in range(num_blocks - 1):
                target_mask_i, context_mask_i = block_mask(num_patches, cfg, block_size, mode, device)
                torch.cat([target_mask, target_mask_i], dim=0)
                torch.cat([context_mask, context_mask_i], dim=0)
            target_masks.append(target_mask)
            context_masks.append(context_mask)
        batch.append((target_masks, context_masks))
    return torch.utils.data.default_collate(batch)

def block_mask(num_patches, cfg, block_size, mode, device):
    """
    :param num_patches: iterable containing the number of patches per dimension

    Returns indcs of mask and its complement
    """
    height, width, depth = num_patches
    h, w, d = block_size

    # Loop to sample masks until one large enough is found
    min_keep = 4 # minimum number of patches to keep
    tries = 0
    timeout = og_timeout = 20
    valid_mask = False
    while not valid_mask:
        # Sample block position
        top = torch.randint(0, height - h + 1, (1,))
        left = torch.randint(0, width - w + 1, (1,))
        back = torch.randint(0, depth - d + 1, (1,))
        mask = torch.zeros((height, width, depth), dtype=torch.int64, device=device)
        mask[top:top+h, left:left+w, back:back+d] = 1
        com_mask = 1 - mask
        mask = torch.nonzero(mask.flatten()).squeeze()
        com_mask = torch.nonzero(com_mask.flatten()).squeeze()

        # If mask too small try again
        valid_mask = len(mask) > min_keep
        if not valid_mask:
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = og_timeout
    return mask, com_mask

def sample_block_size(num_patches, cfg, mode='context'):
    """
    spatial_frac: range that spatial fraction is sampled from
    redshift_frac: range that redshift fraction is sampled from
    aspect_scale: range that x and y ratio is sampled from
    mode: whether to use short or long range sampling parameters

    Helper to sample block mask dimensions
    """
    max_dims = num_patches
    spatial_aspect_range = cfg.spatial_aspect
    match mode:
        case 'short':
            spatial_frac_range = cfg.short_spatial_frac
            redshift_frac_range = cfg.short_redshift_frac
        case 'long':
            spatial_frac_range = cfg.long_spatial_frac
            redshift_frac_range = cfg.long_redshift_frac

    def sample_aspect_ratio(min, max):
        # Sample a single aspect-ratio, ensuring that both dimensions are equally 
        # likely to be scaled up or down
        if torch.randint(0, 2, (1,)) == 0:
            max = 1.
        else:
            min = 1.
        return (min - max) * torch.rand(1,) + max

    def sample_ratio(min, max):
        return (min - max) * torch.rand(1,) + max

    spatial_frac = sample_ratio(*spatial_frac_range) if type(spatial_frac_range) != int else spatial_frac_range
    redshift_frac = sample_ratio(*redshift_frac_range) if type(redshift_frac_range) != int else redshift_frac_range
    while True:
        # Sample ratios such that all dimensions are restricted to within a unit cube
        spatial_aspect = sample_aspect_ratio(*spatial_aspect_range)

        h = math.sqrt(spatial_frac * spatial_aspect)
        w = math.sqrt(spatial_frac / spatial_aspect)
        d = redshift_frac
        dims = [h, w, d]
        dim_outside = [dim > 1. for dim in dims]
        if not any(dim_outside): break

    # Scale unit cube dimensions to number of patches
    for i, dim in enumerate(dims):
        dims[i] = round(dim * max_dims[i])
    
    return dims


def gather_tokens(x, mask):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param mask: tensor of shape [B, K] containing indices of K patches in [N] to keep
    """
    mask_keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
    return torch.gather(x, dim=1, index=mask_keep)