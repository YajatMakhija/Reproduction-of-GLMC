"""Vision Transformer for Generalized Linear Mode Connectivity (CIFAR-10).

Two model forms live here:

* ``ViT`` — a standard pre-norm ViT with **LayerNorm** and biases (QKV is
  bias-free). This is the model you train and load.

* ``ReparamViT`` — the internal canonical form with **parameter-free RMSNorm**
  on a mean-centered residual stream. ``reparameterize(vit)`` converts a trained
  ``ViT`` into this form *function-preservingly*; weight matching and the merger
  operate on it. RMSNorm's rotation invariance is what makes the residual stream
  orthogonally symmetric, which the alignment relies on.

Every ``ReparamViT`` sub-module takes an optional ``coeff``: with ``coeff=None``
it behaves as an ordinary model; the merger swaps sub-modules for coeff-aware
mergers that interpolate two models at the given coefficient.
"""

import copy

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class RMSNorm(nn.Module):
    """Parameter-free RMS normalization with an optional additive offset.

    No learnable scale -> ``RMSNorm(Ox) = O·RMSNorm(x)`` for orthogonal ``O``,
    i.e. the residual stream is invariant to a rotation of its basis.
    """

    def __init__(self, dim, eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.offset = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        out = x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps)
        return out if self.offset is None else out + self.offset


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # bias-free (attention circuit)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(dropout))

    def forward(self, x, coeff=None):
        qkv = (self.to_qkv(x) if coeff is None else self.to_qkv(x, coeff)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        out = self.to_out[0](out) if coeff is None else self.to_out[0](out, coeff)
        return self.to_out[1](out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),   # net[0]
            nn.GELU(),                               # net[1]
            nn.Dropout(dropout),                     # net[2]
            nn.Linear(hidden_dim, dim, bias=bias),   # net[3]
            nn.Dropout(dropout),                     # net[4]
        )

    def forward(self, x, coeff=None):
        return self.net(x) if coeff is None else self.net(x, coeff)


def _block(norm_cls, dim, heads, dim_head, mlp_dim, dropout, bias):
    return nn.ModuleList([
        norm_cls(dim),                                                    # [0] pre-attn norm
        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, bias=bias),  # [1]
        norm_cls(dim),                                                    # [2] pre-ff norm
        FeedForward(dim, mlp_dim, dropout=dropout, bias=bias),            # [3]
    ])


class Transformer(nn.Module):
    """Pre-norm transformer. norm_cls is LayerNorm (ViT) or RMSNorm (ReparamViT)."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, bias, norm_cls):
        super().__init__()
        self.layers = nn.ModuleList([_block(norm_cls, dim, heads, dim_head, mlp_dim, dropout, bias)
                                     for _ in range(depth)])

    def forward(self, x, coeff=None):
        for norm_attn, attn, norm_ff, ff in self.layers:
            if coeff is None:
                x = attn(norm_attn(x)) + x
                x = ff(norm_ff(x)) + x
            else:
                x = attn(norm_attn(x, coeff), coeff) + x
                x = ff(norm_ff(x, coeff), coeff) + x
        return x


class _BaseViT(nn.Module):
    def _init_common(self, image_size, patch_size, num_classes, dim, depth, heads,
                     mlp_dim, channels, dim_head, dropout, bias, norm_cls):
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        self.patch_dim = channels * ph * pw
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw),
            norm_cls(self.patch_dim),                    # [1] patch norm
            nn.Linear(self.patch_dim, dim, bias=bias),   # [2]
            nn.Dropout(dropout),                         # [3]
        )
        self.pos_embedding = nn.Embedding((ih // ph) * (iw // pw), dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, bias, norm_cls)
        self.final_norm = norm_cls(dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes, bias=bias)

    def forward(self, img, coeff=None):
        device = img.device
        x = self.to_patch_embedding(img) if coeff is None else self.to_patch_embedding(img, coeff)
        b, n, _ = x.shape
        pos = torch.arange(n, device=device).unsqueeze(0).expand(b, n)
        x = x + (self.pos_embedding(pos) if coeff is None else self.pos_embedding(pos, coeff))
        x = self.transformer(x, coeff)
        x = self.final_norm(x) if coeff is None else self.final_norm(x, coeff)
        x_all = x
        x = x.mean(dim=1)
        x = self.to_latent(x)
        logits = self.linear_head(x) if coeff is None else self.linear_head(x, coeff)
        return logits, x_all


class ViT(_BaseViT):
    """Standard LayerNorm ViT with biases (QKV bias-free), mean pooling."""

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dim_head=64, dropout=0.2, bias=True):
        super().__init__()
        self._init_common(image_size, patch_size, num_classes, dim, depth, heads,
                          mlp_dim, channels, dim_head, dropout, bias, nn.LayerNorm)


class ReparamViT(_BaseViT):
    """Canonical form: parameter-free RMSNorm on a centered residual stream.
    Built by ``reparameterize``; consumed by weight matching and the merger."""

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        # RMSNorm blocks/final norm carry an offset (folded LayerNorm bias); the
        # patch norm is replaced with the original LayerNorm during reparameterize.
        self._init_common(image_size, patch_size, num_classes, dim, depth, heads,
                          mlp_dim, channels, dim_head, dropout, bias,
                          lambda d: RMSNorm(d, bias=True))


def build_vit(cfg, num_classes, image_size=32):
    return ViT(image_size=image_size, patch_size=cfg["patch_size"], num_classes=num_classes,
               channels=3, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
               mlp_dim=cfg["mlp_dim"], dim_head=cfg["dim"] // cfg["heads"],
               dropout=cfg.get("dropout", 0.2), bias=cfg.get("bias", True))


@torch.no_grad()
def reparameterize(vit: ViT, num_classes, image_size=32):
    """LayerNorm ``ViT`` -> function-preserving ``ReparamViT`` (RMSNorm form).

    Fold each LayerNorm scale gamma into the following reader linear, mean-subtract
    the residual writers (weights and biases) so the stream is centered, and store
    each LayerNorm bias beta/gamma as the RMSNorm offset. Uses the identity
    ``LayerNorm(z) = RMSNorm(zM)·gamma + beta`` with ``M = I - (1/d)11^T``.
    """
    m = copy.deepcopy(vit)
    dim = m.linear_head.weight.shape[1]
    dev = m.linear_head.weight.device
    M = torch.eye(dim, device=dev) - torch.ones(dim, dim, device=dev) / dim

    def fold_scale(ln, reader_weight):
        g = ln.weight.data
        reader_weight.data.copy_(reader_weight.data * g.unsqueeze(0))
        return (ln.bias.data / g).clone()

    offsets = {}
    for i, layer in enumerate(m.transformer.layers):
        offsets[f"attn_{i}"] = fold_scale(layer[0], layer[1].to_qkv.weight)
        offsets[f"ff_{i}"] = fold_scale(layer[2], layer[3].net[0].weight)
    offsets["final"] = fold_scale(m.final_norm, m.linear_head.weight)

    def ms_bias(linear):
        if linear.bias is not None:
            linear.bias.data.copy_(linear.bias.data @ M)

    m.to_patch_embedding[2].weight.data.copy_(M @ m.to_patch_embedding[2].weight.data)
    ms_bias(m.to_patch_embedding[2])
    m.pos_embedding.weight.data.copy_(m.pos_embedding.weight.data @ M)
    for layer in m.transformer.layers:
        layer[1].to_out[0].weight.data.copy_(M @ layer[1].to_out[0].weight.data)
        layer[3].net[3].weight.data.copy_(M @ layer[3].net[3].weight.data)
        ms_bias(layer[1].to_out[0])
        ms_bias(layer[3].net[3])

    r = ReparamViT(image_size=image_size, patch_size=_patch_from(m, image_size),
                   num_classes=num_classes, channels=3, dim=dim,
                   depth=len(m.transformer.layers), heads=m.transformer.layers[0][1].heads,
                   mlp_dim=m.transformer.layers[0][3].net[0].weight.shape[0],
                   dim_head=dim // m.transformer.layers[0][1].heads,
                   dropout=0.0, bias=(m.linear_head.bias is not None)).to(dev)

    def cp(dst, src):
        dst.weight.data.copy_(src.weight.data)
        if getattr(src, "bias", None) is not None and getattr(dst, "bias", None) is not None:
            dst.bias.data.copy_(src.bias.data)

    cp(r.to_patch_embedding[2], m.to_patch_embedding[2])
    r.pos_embedding.weight.data.copy_(m.pos_embedding.weight.data)
    cp(r.linear_head, m.linear_head)
    for i, layer in enumerate(m.transformer.layers):
        r.transformer.layers[i][1].to_qkv.weight.data.copy_(layer[1].to_qkv.weight.data)
        cp(r.transformer.layers[i][1].to_out[0], layer[1].to_out[0])
        cp(r.transformer.layers[i][3].net[0], layer[3].net[0])
        cp(r.transformer.layers[i][3].net[3], layer[3].net[3])
        r.transformer.layers[i][0].offset.data.copy_(offsets[f"attn_{i}"])
        r.transformer.layers[i][2].offset.data.copy_(offsets[f"ff_{i}"])
    # patch-embed norm stays a LayerNorm (off the residual stream)
    r.to_patch_embedding[1] = copy.deepcopy(m.to_patch_embedding[1])
    r.final_norm.offset.data.copy_(offsets["final"])
    return r


def _patch_from(m, image_size):
    # recover patch size from patch_dim (channels=3)
    return int((m.patch_dim // 3) ** 0.5)
