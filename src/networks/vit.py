"""Modified from github.com/facebookresearch/DiT/blob/main/models.py""" 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention, Mlp
from torch.utils.checkpoint import checkpoint

class ViT(nn.Module):
    """
    A vision transformer network.
    """

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        in_channels, *axis_sizes = cfg.in_shape
        dim = cfg.hidden_dim

        # check consistency of arguments
        check_shapes(cfg)
        
        # embedding layer
        patch_dim = math.prod(cfg.patch_shape) * in_channels
        self.embedding = nn.Linear(patch_dim, dim)

        # position encoding
        fourier_dim = dim // 6 # sin/cos features for each dime
        w = torch.arange(fourier_dim) / (fourier_dim - 1)
        w = (1. / (10_000 ** w)).repeat(3)
        self.pos_encoding_freqs = nn.Parameter(
            w.log() if cfg.learn_pos_encoding else w, requires_grad=cfg.learn_pos_encoding
        )
        self.num_patches = [s // p for s, p in zip(axis_sizes, cfg.patch_shape)]
        for i, n in enumerate(self.num_patches): # axis values for each dim
            self.register_buffer(f'grid_{i}', torch.arange(n)*(2*math.pi/n))

        # transformer stack
        self.blocks = nn.ModuleList([
            Block(
                dim, cfg.num_heads, mlp_ratio=cfg.mlp_ratio, mlp_drop=cfg.mlp_drop,
                checkpoint_grads=cfg.checkpoint_grads, attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop
            ) for _ in range(cfg.depth)
        ])

        # output head
        self.head = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(cfg.mlp_drop),
            nn.Linear(dim, cfg.out_channels)
        )
        self.out_act = getattr(F, cfg.out_act) if cfg.out_act else nn.Identity()

        # masking
        if cfg.mask_frac > 0:
            assert cfg.mask_frac < 1
            self.mask_token = nn.Parameter(torch.randn(dim))

    def pos_encoding(self): # TODO: Simplify for fixed dim=3
        grids = [getattr(self, f'grid_{i}') for i in range(3)]
        coords = torch.meshgrid(*grids, indexing='ij')

        if self.cfg.learn_pos_encoding:
            freqs = self.pos_encoding_freqs.exp().chunk(3)
        else:
            freqs = self.pos_encoding_freqs.chunk(3)

        features = [
            trig_fn(x.flatten()[:,None] * w[None, :])
            for (x, w) in zip(coords, freqs) for trig_fn in (torch.sin, torch.cos)
        ]
        return torch.cat(features, dim=1)

    def forward(self, x, mask=False):
        """
        Forward pass of ViT.
        x   : tensor of spatial inputs with shape (B, C, *axis_sizes)
        mask: whether or not mask patches (for self supervision).
        """

        # patchify input and embed
        x = self.to_patches(x) # (B, T, D), with T = prod(num_patches)
        x = self.embedding(x)
        if mask:
            self.random_mask_patches(x)
        x = x + self.pos_encoding() # TODO: Check whether masked tokens should really get a position embedding

        # process patches with transformer blocks
        for block in self.blocks:
            x = block(x)

        # aggregate patch features
        x = torch.mean(x, axis=1) # (B, D)

        # apply task head
        x = self.head(x) # (B, Z)
        
        return self.out_act(x) 

    def to_patches(self, x):
        x = rearrange(
            x, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)',
            **dict(zip(('p1', 'p2', 'p3'), self.cfg.patch_shape))
        )
        return x

    def random_mask_patches(self, x):
        """
        Masks x by randomly selecting patches in each batch and replacing their
        embedding with `self.mask_token`. The number of patches to mask is 
        determined by the `self.cfg.mask_frac` option.
        """
        if not self.cfg.mask_frac:
            print("WARNING: Option `mask_frac` is zero. No masking will be applied.")
            return x

        B, T = x.size(0), math.prod(self.num_patches)
        num_masked = int(self.cfg.mask_frac * T)
        full_mask = repeat(self.mask_token, 'd -> b t d', b=B, t=T)
        mask_idcs = torch.rand(B, T, device=x.device).topk(k=num_masked, dim=-1).indices
        mask_map = torch.zeros((B, T), device=x.device).scatter_(-1, mask_idcs, 1).bool()

        return torch.where(mask_map[..., None], full_mask, x)

class Block(nn.Module):
    def __init__(
            self, hidden_size, num_heads, mlp_ratio=4.0, mlp_drop=0., checkpoint_grads=False,
            **attn_kwargs
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **attn_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu,
            drop=mlp_drop
        )
        self.checkpoint_grads = checkpoint_grads

    def forward(self, x):
        if self.checkpoint_grads:
            x = x + checkpoint(self.attn, self.norm1(x), use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
def check_shapes(cfg):
    for i, (s, p) in enumerate(zip(cfg.in_shape[1:], cfg.patch_shape)):
        assert not s % p, \
            f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
    assert not cfg.hidden_dim % 6, \
        f"Hidden dim should be divisible by 6 (for fourier position embeddings)"    