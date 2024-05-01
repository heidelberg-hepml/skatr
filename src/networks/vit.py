"""Modified from github.com/facebookresearch/DiT/blob/main/models.py""" 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp


class ViT(nn.Module):
    """
    A vision transformer network.
    """

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        in_channels, *axis_sizes = cfg.in_shape

        # check shapes
        for i, (s, p) in enumerate(zip(axis_sizes, cfg.patch_shape)):
            assert not s % p, \
                f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
        assert not cfg.hidden_dim % 6, \
            f"Hidden dim should be divisible by 6 (for fourier position embeddings)"
        
        # initialize x,t,c embeddings
        patch_dim = math.prod(cfg.patch_shape) * in_channels
        self.x_embedder = nn.Linear(patch_dim, cfg.hidden_dim)
        self.num_patches = [s // p for s, p in zip(axis_sizes, cfg.patch_shape)]
        
        # initialize fourier frequencies for position embeddings
        fourier_dim = cfg.hidden_dim // 6
        w = torch.arange(fourier_dim) / (fourier_dim - 1)
        w = 1. / (10_000 ** w)
        w = w.repeat(3)
        self.pos_encoding_freqs = nn.Parameter(
            w.log() if cfg.learn_pos_encoding else w,
            requires_grad=cfg.learn_pos_encoding
        )

        # initialize coordinate grids for position embeddings
        for i, n in enumerate(self.num_patches):
            self.register_buffer(f'grid_{i}', torch.arange(n)*(2*math.pi/n))

        # initialize transformer stack
        self.blocks = nn.ModuleList([
            Block(
                cfg.hidden_dim, cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop,
            ) for _ in range(cfg.depth)
        ])

        # initialize output layer
        # if cfg.final_conv: # TODO: Is final conv sensible for classification?
        #     final_conv_channels = cfg.final_conv_channels or in_channels
        #     self.final_layer = FinalLayer(cfg.hidden_dim, cfg.patch_shape, final_conv_channels)
        #     self.conv_layer = nn.Conv3d(final_conv_channels, in_channels, kernel_size=3, padding=1)
        # else:
        self.final_layer = FinalProj(cfg.hidden_dim, cfg.patch_shape, cfg.out_channels, cfg.out_act)

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


    def forward(self, x):
        """
        Forward pass of ViT.
        x: (B, C, *axis_sizes) tensor of spatial inputs
        """
        
        x = self.to_patches(x)                       # (B, T, D), where T = prod(num_patches)
        x = self.x_embedder(x) + self.pos_encoding() # (B, T, D)

        N = (len(self.blocks)+1)//2 - 1 # for long skips
        residuals = []                  # for long skips
        for i, block in enumerate(self.blocks):
            x = block(x) # (B, T, D)
            if self.cfg.long_skips:
                if i < N:
                    residuals.append(x)
                elif i >= len(self.blocks)-N:
                    x = x + residuals.pop()

        x = self.final_layer(x) # (B, T, prod(patch_shape) * out_channels)

        return x

    def to_patches(self, x):
        x = rearrange(
            x, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)',
            **dict(zip(('p1', 'p2', 'p3'), self.cfg.patch_shape))
        )
        return x
                 
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FinalProj(nn.Module):
    # TODO: get rid of class and just define layers in ViT.__init__
    def __init__(self, hidden_dim, patch_shape, out_channels, act=None):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_channels)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.act = getattr(F, act) if act else nn.Identity()

    def forward(self, x):
        x = self.norm1(x)
        x = self.linear1(x) # TODO: Replace with MLP?
        x = torch.mean(x, axis=1)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.act(x)
        return x