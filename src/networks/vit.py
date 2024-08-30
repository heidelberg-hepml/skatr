import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from hydra.utils import instantiate
from torch.utils.checkpoint import checkpoint

from src.utils import masks

class ViT(nn.Module):
    """
    A vision transformer network.
    """

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg
        self.patch_shape = cfg.patch_shape
        in_channels, *axis_sizes = cfg.in_shape
        dim = cfg.hidden_dim

        # check consistency of arguments
        check_shapes(cfg)
        
        # embedding layer
        self.patch_dim = math.prod(cfg.patch_shape) * in_channels
        self.embedding = nn.Linear(self.patch_dim, dim)

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

        # optionally initialize a task head, input pooling, or mask token
        if cfg.use_head:
            self.init_head(cfg.head)
        if cfg.adapt_res:
            self.init_adaptor(cfg.adaptor)
        if self.cfg.use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(dim))

    def init_head(self, cfg):
        self.head = instantiate(cfg)

    def init_adaptor(self, channels, downsample_factor, replace_embedding):
        self.adaptor = nn.Sequential(
            nn.Conv3d(1,  channels, downsample_factor, downsample_factor), nn.ReLU()
        )
        if replace_embedding:
            self.embedding = nn.Linear(channels * self.patch_dim, self.cfg.hidden_dim)
        else:
            self.extra_proj = nn.Linear(channels * self.patch_dim, self.patch_dim)

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
        :param x   : tensor of spatial inputs with shape (batch_size, channels, *axis_sizes)
        :param mask: a tensor of patch indices that should be masked out of `x`.
        """

        if hasattr(self, 'conv'):
            x = self.conv(x)
        if hasattr(self, 'adaptor'):
            x = self.adaptor(x)
            
        # patchify input
        # x -> (batch_size, number_of_patches, voxels_per_patch)
        x = self.to_patches(x)
        
        # embed
        # x -> (batch_size, number_of_patches, embedding_dim)
        if hasattr(self, 'extra_proj'):
            x = self.extra_proj(x)
        x = self.embedding(x)

        # apply mask and position encoding
        if self.cfg.use_mask_token:
            if mask is not None:
                x = self.apply_mask_tokens(x, mask)
            x = x + self.pos_encoding()
        else:
            # x -> (batch_size, number_of_masked_patches, embedding_dim)
            x = x + self.pos_encoding()
            if mask is not None:
                x = masks.gather_tokens(x, mask)
        
        # process patches with transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)

        if hasattr(self, 'head'):
            # aggregate patch features and apply task head
            # x -> (batch_size, out_channels)
            x = torch.mean(x, axis=1)
            x = self.head(x)

        return x

    def to_patches(self, x):
        x = rearrange(
            x, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)',
            **dict(zip(('p1', 'p2', 'p3'), self.patch_shape))
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
    

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

def check_shapes(cfg):
    for i, (s, p) in enumerate(zip(cfg.in_shape[1:], cfg.patch_shape)):
        assert not s % p, \
            f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
    assert not cfg.hidden_dim % 6, \
        f"Hidden dim should be divisible by 6 (for fourier position embeddings)"