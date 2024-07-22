import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from hydra.utils import instantiate
from timm.models.vision_transformer import Attention, Mlp
from torch.utils.checkpoint import checkpoint

from src.utils import masks

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
        fourier_dim = dim // 6 # sin/cos features for each dim
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

        # norm layer
        self.out_norm = nn.LayerNorm(dim, eps=1e-6)

        # output head
        if cfg.use_head:
            self.head = instantiate(cfg.head)

        # masking
        if self.cfg.use_mask_token:
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

    def forward(self, x, mask=None):
        """
        Forward pass of ViT.
        :param x   : tensor of spatial inputs with shape (B, C, *axis_sizes)
        :param mask: a tensor of patch indices that should be masked out of `x`.
        """

        # patchify input and embed
        x = self.to_patches(x) # (B, T, D), with T = prod(num_patches)
        x = self.embedding(x)

        # apply mask and position encoding
        if self.cfg.use_mask_token:
            if mask is not None:
                x = self.apply_mask_tokens(x, mask)
            x = x + self.pos_encoding()
        else:
            x = x + self.pos_encoding()
            if mask is not None:
                x = masks.gather_tokens(x, mask)
        
        # process patches with transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)

        if self.cfg.use_head:
            # aggregate patch features and apply task head
            x = torch.mean(x, axis=1) # (B, D)
            x = self.head(x) # (B, Z)

        return x

    def to_patches(self, x):
        x = rearrange(
            x, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)',
            **dict(zip(('p1', 'p2', 'p3'), self.cfg.patch_shape))
        )
        return x

    def apply_mask_tokens(self, x, mask_idcs):
        """
        Replaces patch embeddings in `x` with the network's mask token at indices speficied by `mask`.

        :param x   : input tensor with shape (B [batch size], T [number of patches], D [embed dim])
        :param mask: tensor with shape (B, T) containing indices in the range [0,T)
        """
        B, T = x.shape[:2]
        full_mask_token = repeat(self.mask_token, 'd -> b t d', b=B, t=T)
        # construct boolean mask
        mask = torch.zeros((B, T), device=x.device).scatter_(-1, mask_idcs, 1).bool()
        return torch.where(mask[..., None], full_mask_token, x)    

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

class PredictorViT(ViT):

    def __init__(self, cfg):

        super().__init__(cfg)

        # override embedding layer # TODO: better way?
        self.embedding = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.in_dim)


    def forward(self, ctx, ctx_mask, tgt_mask):
        """
        :param ctx: tokens from context block
        :param ctx_mask: mask corresponding to context block (for pos encoding)
        :param tgt_mask: mask corresponding to target block (for pos encoding)
        """
        
        B, N_ctx, D = ctx.shape # batch size, num context patches, context dim
        T = math.prod(self.num_patches) # total patches (before masking)
        
        pos_encoding = repeat(self.pos_encoding(), 't d -> b t d', b=B)

        # embed context tokens to own hidden dim
        ctx = self.embedding(ctx)
        ctx = ctx + masks.gather_tokens(pos_encoding, ctx_mask) # TODO: Correct? or different pos encoding needed?

        # prepare target prediction tokens
        tgt = repeat(self.mask_token, 'd -> b t d', b=B, t=T) # repeat to full shape
        tgt = tgt + pos_encoding # add position encodings
        tgt = masks.gather_tokens(tgt, tgt_mask) # only keep tokens in target block

        # concatenate
        prd = torch.cat([ctx, tgt], dim=1)
        
        # process patches with transformer blocks
        for block in self.blocks:
            prd = block(prd)
        prd = self.out_norm(prd)

        prd = prd[:, N_ctx:] # select output tokens in target block
        prd = self.out_proj(prd) # project back to full dimensions

        return prd

def check_shapes(cfg):
    for i, (s, p) in enumerate(zip(cfg.in_shape[1:], cfg.patch_shape)):
        assert not s % p, \
            f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
    assert not cfg.hidden_dim % 6, \
        f"Hidden dim should be divisible by 6 (for fourier position embeddings)"    