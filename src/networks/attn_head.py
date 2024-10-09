# adapted from https://github.com/facebookresearch/jepa/blob/main/src/models/attentive_pooler.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.pooler = AttentivePooler(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            norm_layer=cfg.norm_layer,
            init_std=cfg.init_std,
            qkv_bias=cfg.qkv_bias,
            complete_block=cfg.complete_block,
            use_proj=cfg.use_proj,
        )
        if cfg.use_act:
            self.act = getattr(F, cfg.act)
        if cfg.use_linear:
            self.linear = nn.Linear(cfg.embed_dim, cfg.out_channels, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        if hasattr(self, "act"):
            x = self.act(x)
        if hasattr(self, "linear"):
            x = self.linear(x)
        return x


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        embed_dim=144,
        num_heads=6,
        mlp_ratio=4.0,
        norm_layer="LayerNorm",
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_proj=True,
    ):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_proj = use_proj
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, use_proj=use_proj
            )

        self.init_std = init_std
        nn.init.trunc_normal_(self.query_token, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        elif self.use_proj:
            rescale(self.cross_attention_block.proj.weight.data, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_token.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=6, qkv_bias=False, use_sdpa=True, use_proj=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.use_sdpa = use_sdpa
        if use_proj:
            self.proj = nn.Linear(dim, dim)

    def forward(self, q, x):
        B, n, C = q.shape
        q = (
            self.q(q)
            .reshape(B, n, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        B, N, C = x.shape
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        if hasattr(self, "proj"):
            q = self.proj(q)

        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer="LayerNorm",
    ):
        super().__init__()
        self.norm1 = getattr(nn, norm_layer)(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = getattr(nn, norm_layer)(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
