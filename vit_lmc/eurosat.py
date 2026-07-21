"""Linear mode connectivity of two ViTs on EuroSAT (GLMC)."""

import copy
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import kornia.augmentation as K
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
import ot
import matplotlib.pyplot as plt


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16

if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu] {_gpu_name}  ({_vram:.0f} GB VRAM)  device={device}")
else:
    print("[gpu] CUDA not available — running on CPU (will be slow)")


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class RMSNorm(nn.Module):
    """Parameter-free RMS norm with optional additive offset (rotation-invariant)."""

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
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
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
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x, coeff=None):
        return self.net(x) if coeff is None else self.net(x, coeff)

def _block(norm_cls, dim, heads, dim_head, mlp_dim, dropout, bias):
    return nn.ModuleList([
        norm_cls(dim),
        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, bias=bias),
        norm_cls(dim),
        FeedForward(dim, mlp_dim, dropout=dropout, bias=bias),
    ])

class Transformer(nn.Module):
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
            norm_cls(self.patch_dim),
            nn.Linear(self.patch_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Embedding((ih // ph) * (iw // pw), dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, bias, norm_cls)
        self.final_norm = norm_cls(dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes, bias=bias)

    def forward(self, img, coeff=None):
        dev = img.device
        x = self.to_patch_embedding(img) if coeff is None else self.to_patch_embedding(img, coeff)
        b, n, _ = x.shape
        pos = torch.arange(n, device=dev).unsqueeze(0).expand(b, n)
        x = x + (self.pos_embedding(pos) if coeff is None else self.pos_embedding(pos, coeff))
        x = self.transformer(x, coeff)
        x = self.final_norm(x) if coeff is None else self.final_norm(x, coeff)
        x_all = x
        x = x.mean(dim=1)
        x = self.to_latent(x)
        logits = self.linear_head(x) if coeff is None else self.linear_head(x, coeff)
        return logits, x_all

class ViT(_BaseViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dim_head=64, dropout=0.2, bias=True):
        super().__init__()
        self._init_common(image_size, patch_size, num_classes, dim, depth, heads,
                          mlp_dim, channels, dim_head, dropout, bias, nn.LayerNorm)

class ReparamViT(_BaseViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        self._init_common(image_size, patch_size, num_classes, dim, depth, heads,
                          mlp_dim, channels, dim_head, dropout, bias,
                          lambda d: RMSNorm(d, bias=True))

def build_vit(cfg, num_classes, image_size=32):
    return ViT(image_size=image_size, patch_size=cfg["patch_size"], num_classes=num_classes,
               channels=3, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
               mlp_dim=cfg["mlp_dim"], dim_head=cfg["dim"] // cfg["heads"],
               dropout=cfg.get("dropout", 0.2), bias=cfg.get("bias", True))

def _patch_from(m, image_size):
    return int((m.patch_dim // 3) ** 0.5)

@torch.no_grad()
def reparameterize(vit, num_classes, image_size=32):
    """LayerNorm ViT -> function-preserving RMSNorm ReparamViT."""
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
    r.to_patch_embedding[1] = copy.deepcopy(m.to_patch_embedding[1])
    r.final_norm.offset.data.copy_(offsets["final"])
    return r


def interpolate(W0, W1, coeff):
    return coeff * W0 + (1 - coeff) * W1

def _make_orthogonal(A):
    U, _, Vt = torch.linalg.svd(A.cpu(), full_matrices=False)
    return (U @ Vt).to(A.device)

def _make_permutation(A):
    row, col = linear_sum_assignment(-A.detach().cpu().numpy())
    P = torch.zeros_like(A)
    P[row, col] = 1
    return P

def project(A, kind):
    """Project A onto a manifold. 'perm' uses a straight-through estimator."""
    if kind == "perm":
        P = _make_permutation(A)
        return P.detach() + (A - A.detach())
    if kind == "ortho":
        return _make_orthogonal(A)
    raise ValueError(kind)

def project_to_attn_circuits(model, heads, dim, layer_i):
    Q, K, V = model.transformer.layers[layer_i][1].to_qkv.weight.data.chunk(3, dim=0)
    Q = rearrange(Q, "(h d) m -> h d m", h=heads, m=dim)
    K = rearrange(K, "(h d) m -> h d m", h=heads, m=dim)
    V = rearrange(V, "(h d) m -> h d m", h=heads, m=dim)
    OUT = rearrange(model.transformer.layers[layer_i][1].to_out[0].weight.data,
                    "m (h d) -> m h d", h=heads, m=dim).permute(1, 2, 0)
    QK = torch.bmm(Q.transpose(1, 2), K)
    OUTV = OUT.transpose(1, 2) @ V
    Q_new = torch.zeros_like(QK); K_new = torch.zeros_like(QK); V_new = torch.zeros_like(QK)
    OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1], device=QK.device)
    for h in range(QK.size(0)):
        eye = torch.eye(QK[h].shape[0], device=QK.device)
        Q_new[h] = QK[h].t();  K_new[h] = eye
        OUT_new[h] = OUTV[h].t();  V_new[h] = eye
    model.transformer.layers[layer_i][1].to_qkv.weight.data = torch.cat(
        (Q_new.reshape(-1, dim), K_new.reshape(-1, dim), V_new.reshape(-1, dim)), dim=0)
    model.transformer.layers[layer_i][1].to_out[0].weight.data = OUT_new.permute(2, 0, 1).reshape(dim, -1)
    return model


def compute_optimal_orthogonal_matrix(t1, t2):
    U, _, Vh = torch.linalg.svd(t2.T @ t1)
    return U @ Vh

def _rotate(x, O):
    return x @ O.t() if x is not None else None

def ortho_residual(model, O):
    model.to_patch_embedding[2].weight.data = O @ model.to_patch_embedding[2].weight.data
    if model.to_patch_embedding[2].bias is not None:
        model.to_patch_embedding[2].bias.data = _rotate(model.to_patch_embedding[2].bias.data, O)
    model.pos_embedding.weight.data = model.pos_embedding.weight.data @ O.t()
    for layer in model.transformer.layers:
        layer[1].to_qkv.weight.data = layer[1].to_qkv.weight.data @ O.t()
        layer[1].to_out[0].weight.data = O @ layer[1].to_out[0].weight.data
        layer[3].net[0].weight.data = layer[3].net[0].weight.data @ O.t()
        layer[3].net[3].weight.data = O @ layer[3].net[3].weight.data
        if layer[1].to_out[0].bias is not None:
            layer[1].to_out[0].bias.data = _rotate(layer[1].to_out[0].bias.data, O)
        if layer[3].net[3].bias is not None:
            layer[3].net[3].bias.data = _rotate(layer[3].net[3].bias.data, O)
        layer[0].offset.data = _rotate(layer[0].offset.data, O)
        layer[2].offset.data = _rotate(layer[2].offset.data, O)
    model.final_norm.offset.data = _rotate(model.final_norm.offset.data, O)
    model.linear_head.weight.data = model.linear_head.weight.data @ O.t()
    return model

def _residual_stack(model0, model1, include_blocks):
    a = [model0.to_patch_embedding[2].weight.data, model0.pos_embedding.weight.data.t(), model0.linear_head.weight.data.t()]
    b = [model1.to_patch_embedding[2].weight.data, model1.pos_embedding.weight.data.t(), model1.linear_head.weight.data.t()]
    if include_blocks:
        for l0, l1 in zip(model0.transformer.layers, model1.transformer.layers):
            a += [l0[1].to_qkv.weight.data.t(), l0[1].to_out[0].weight.data, l0[3].net[0].weight.data.t(), l0[3].net[3].weight.data]
            b += [l1[1].to_qkv.weight.data.t(), l1[1].to_out[0].weight.data, l1[3].net[0].weight.data.t(), l1[3].net[3].weight.data]
    a = [t / t.shape[1] ** 0.5 for t in a]
    b = [t / t.shape[1] ** 0.5 for t in b]
    return torch.cat(a, dim=1).t(), torch.cat(b, dim=1).t()

def otify(cost):
    n = cost.shape[0]
    return ot.emd(torch.ones(n) / n, torch.ones(n) / n, cost) * n

def _permute(A, P):
    return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])

def get_cost_heads(t0, t1, heads):
    cost = torch.zeros((heads, heads))
    for i in range(heads):
        for j in range(heads):
            cost[i, j] = torch.sqrt(torch.sum((t0[i] - t1[j]) ** 2))
    return cost

def _head_circuits(qkv_model, out_model, heads, dim, layer_i):
    Q, K, V = qkv_model.transformer.layers[layer_i][1].to_qkv.weight.data.chunk(3, dim=0)
    Q = rearrange(Q, "(h d) m -> h d m", h=heads, m=dim)
    K = rearrange(K, "(h d) m -> h d m", h=heads, m=dim)
    V = rearrange(V, "(h d) m -> h d m", h=heads, m=dim)
    OUT = rearrange(out_model.transformer.layers[layer_i][1].to_out[0].weight.data,
                    "m (h d) -> m h d", h=heads, m=dim).permute(1, 2, 0)
    return torch.bmm(Q.transpose(1, 2), K), OUT.transpose(1, 2) @ V

def permute_heads(model, P, heads, dim, layer_i):
    Q, K, V = model.transformer.layers[layer_i][1].to_qkv.weight.data.chunk(3, dim=0)
    Q = rearrange(Q, "(h d) m -> h d m", h=heads, m=dim)
    K = rearrange(K, "(h d) m -> h d m", h=heads, m=dim)
    V = rearrange(V, "(h d) m -> h d m", h=heads, m=dim)
    OUT = rearrange(model.transformer.layers[layer_i][1].to_out[0].weight.data,
                    "m (h d) -> m h d", h=heads, m=dim).permute(1, 2, 0)
    Q, K, V, OUT = _permute(Q, P), _permute(K, P), _permute(V, P), _permute(OUT, P)
    head_dim = Q.shape[1]
    QK = torch.bmm(Q.transpose(1, 2), K)
    OUTV = OUT.transpose(1, 2) @ V

    def split(A):
        U, S, Vh = torch.linalg.svd(A, full_matrices=True)
        return U[:, :head_dim] @ torch.diag(S[:head_dim]), Vh[:head_dim, :]

    for h in range(QK.size(0)):
        US, Vr = split(QK[h]);  Q[h] = US.t();  K[h] = Vr
        US, Vr = split(OUTV[h]); OUT[h] = US.t(); V[h] = Vr
    model.transformer.layers[layer_i][1].to_qkv.weight.data = torch.cat(
        (Q.reshape(-1, dim), K.reshape(-1, dim), V.reshape(-1, dim)), dim=0)
    model.transformer.layers[layer_i][1].to_out[0].weight.data = OUT.permute(2, 0, 1).reshape(dim, -1)
    return model

def permute_mlp(model, P, layer_i):
    net = model.transformer.layers[layer_i][3].net
    net[0].weight.data = P @ net[0].weight.data
    net[3].weight.data = net[3].weight.data @ P.t()
    if net[0].bias is not None:
        net[0].bias.data = P @ net[0].bias.data
    return model

def weight_matching(model0, model1, heads, iterations=15):
    for it in range(iterations):
        R0, R1 = _residual_stack(model0, model1, include_blocks=(it > 0))
        ortho_residual(model1, compute_optimal_orthogonal_matrix(R0, R1).t())
        for layer_i in range(len(model1.transformer.layers)):
            dim = model1.transformer.layers[layer_i][1].to_qkv.weight.data.shape[1]
            QK0, OUTV0 = _head_circuits(model0, model1, heads, dim, layer_i)
            QK1, OUTV1 = _head_circuits(model1, model1, heads, dim, layer_i)
            cost = get_cost_heads(QK0, QK1, heads) + get_cost_heads(OUTV0, OUTV1, heads)
            permute_heads(model1, otify(cost).to(QK0.device), heads, dim, layer_i)
            ff0 = torch.cat((model0.transformer.layers[layer_i][3].net[0].weight.data,
                             model0.transformer.layers[layer_i][3].net[3].weight.data.t()), dim=1)
            ff1 = torch.cat((model1.transformer.layers[layer_i][3].net[0].weight.data,
                             model1.transformer.layers[layer_i][3].net[3].weight.data.t()), dim=1)
            cost_ff = torch.cdist(ff0 / torch.norm(ff0, dim=-1, keepdim=True),
                                  ff1 / torch.norm(ff1, dim=-1, keepdim=True), p=1).cpu()
            permute_mlp(model1, otify(cost_ff).to(ff0.device), layer_i)
    return model1


class LinearMerger(nn.Module):
    def __init__(self, linear0, linear1):
        super().__init__()
        self.register_buffer("w0", linear0.weight.data.t().clone().contiguous())
        self.register_buffer("w1", linear1.weight.data.t().clone().contiguous())
        self.has_bias = linear0.bias is not None and linear1.bias is not None
        if self.has_bias:
            self.register_buffer("b0", linear0.bias.data.clone())
            self.register_buffer("b1", linear1.bias.data.clone())
        self.P_in = self.P_out = None
        self.nf = self.w0.shape[1]

    def set_P_in(self, P):  self.P_in = P
    def set_P_out(self, P): self.P_out = P

    def forward(self, x, coeff=0.5):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.w0, self.P_in @ self.w1 @ self.P_out, coeff)
        x = torch.matmul(x.view(-1, x.size(-1)), weight)
        if self.has_bias:
            x = x + interpolate(self.b0, self.b1 @ self.P_out, coeff)
        return x.view(size_out)

class EmbeddingMerger(nn.Module):
    def __init__(self, embedding0, embedding1):
        super().__init__()
        self.embedding_0 = copy.deepcopy(embedding0)
        self.embedding_1 = copy.deepcopy(embedding1)
        for p in self.parameters():
            p.requires_grad = False
        self.P = None

    def set_P(self, P): self.P = P

    def forward(self, x, coeff=0.5):
        return interpolate(self.embedding_0(x), self.embedding_1(x) @ self.P, coeff)

class RMSMerger(nn.Module):
    def __init__(self, rmsnorm0, rmsnorm1, dim):
        super().__init__()
        off0, off1 = rmsnorm0.offset, rmsnorm1.offset
        self.has_bias = off0 is not None and off1 is not None
        if self.has_bias:
            self.register_buffer("bias_0", off0.data.clone())
            self.register_buffer("bias_1", off1.data.clone())
        self.norm = RMSNorm(dim, eps=rmsnorm0.eps)
        self.P = None

    def set_P(self, P): self.P = P

    def forward(self, x, coeff=0.5):
        x = self.norm(x)
        if self.has_bias:
            return x + interpolate(self.bias_0, self.bias_1 @ self.P.t(), coeff)
        return x

class LayerNormMerger(nn.Module):
    def __init__(self, ln0, ln1):
        super().__init__()
        self.eps = ln0.eps
        for n, m in (("w0", ln0.weight), ("b0", ln0.bias), ("w1", ln1.weight), ("b1", ln1.bias)):
            self.register_buffer(n, m.data.clone())

    def forward(self, x, coeff=0.5):
        xn = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + self.eps)
        return interpolate(self.w0, self.w1, coeff) * xn + interpolate(self.b0, self.b1, coeff)

class PatchEmbeddingMerger(nn.Module):
    def __init__(self, patch_embedding):
        super().__init__()
        self.patch_embedding = patch_embedding

    def __getitem__(self, idx): return self.patch_embedding[idx]

    def forward(self, x, coeff):
        x = self.patch_embedding[0](x)
        x = self.patch_embedding[1](x, coeff)
        x = self.patch_embedding[2](x, coeff)
        return self.patch_embedding[3](x)

class FeedForwardMerger(nn.Module):
    def __init__(self, feedforward):
        super().__init__()
        self.feedforward = feedforward

    def __getitem__(self, idx): return self.feedforward[idx]

    def forward(self, x, coeff):
        x = self.feedforward[0](x, coeff)
        x = self.feedforward[1](x)
        x = self.feedforward[2](x)
        x = self.feedforward[3](x, coeff)
        return self.feedforward[4](x)

class AttnQKVMerger(nn.Module):
    def __init__(self, qkv0, qkv1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("w0", qkv0.weight.data.t().clone().contiguous())
        self.register_buffer("w1", qkv1.weight.data.t().clone().contiguous())
        self.P_in = self.P_out = None
        self.nf = self.w0.shape[1]
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def set_P_in(self, P):  self.P_in = P
    def set_P_out(self, P): self.P_out = P

    def _permute_heads(self, weight, P):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        Q, K, V = weight.t().chunk(3, dim=0)
        Q = Q @ self.P_in.t()
        Q = rearrange(Q, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim)
        K = rearrange(K, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim)
        V = rearrange(V, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim)
        Q = torch.bmm(Q.transpose(1, 2), self.P_in.t().expand(self.num_heads, -1, -1)).transpose(1, 2)
        Q, K, V = permute(Q, P), permute(K, P), permute(V, P)
        return torch.cat((Q.reshape(-1, self.embed_dim), K.reshape(-1, self.embed_dim),
                          V.reshape(-1, self.embed_dim)), dim=0).t()

    def forward(self, x, coeff=0.5):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.w0, self._permute_heads(self.w1, self.P_out), coeff)
        return torch.mm(x.view(-1, x.size(-1)), weight).view(size_out)

class AttnOutMerger(nn.Module):
    def __init__(self, out0, out1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("w0", out0.weight.data.t().clone().contiguous())
        self.register_buffer("w1", out1.weight.data.t().clone().contiguous())
        self.has_bias = out0.bias is not None and out1.bias is not None
        if self.has_bias:
            self.register_buffer("b0", out0.bias.data.clone())
            self.register_buffer("b1", out1.bias.data.clone())
        self.P_in = self.P_out = None
        self.nf = self.w0.shape[1]
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def set_P_in(self, P):  self.P_in = P
    def set_P_out(self, P): self.P_out = P

    def _permute_heads(self, weight, P):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        OUT = rearrange(weight.t(), "m (h d) -> m h d", h=self.num_heads, m=self.embed_dim).permute(1, 2, 0)
        OUT = (OUT.transpose(1, 2) @ self.P_out.expand(self.num_heads, -1, -1)).transpose(1, 2)
        OUT = permute(OUT, P)
        return OUT.permute(2, 0, 1).reshape(self.embed_dim, -1).t()

    def forward(self, x, coeff=0.5):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.w0, self._permute_heads(self.w1 @ self.P_out, self.P_in), coeff)
        x = torch.mm(x.view(-1, x.size(-1)), weight)
        if self.has_bias:
            x = x + interpolate(self.b0, self.b1 @ self.P_out, coeff)
        return x.view(size_out)

class ViTMerger(nn.Module):
    def __init__(self, model0, model1, num_heads, device="cpu", permutations_only=False):
        super().__init__()
        model0, model1 = model0.eval(), model1.eval()
        embed_dim = model0.pos_embedding.weight.shape[1]
        assert embed_dim == model1.pos_embedding.weight.shape[1], "equal-width models only"
        patch_dim = model0.to_patch_embedding[2].weight.shape[1]
        n_layers = len(model0.transformer.layers)
        self.permutations_only = bool(permutations_only)

        for i in range(n_layers):
            project_to_attn_circuits(model0, num_heads, embed_dim, i)
            project_to_attn_circuits(model1, num_heads, embed_dim, i)

        self.proj = nn.ParameterDict({"residual": nn.Parameter(torch.eye(embed_dim))})
        for i in range(n_layers):
            self.proj[f"attention_heads_{i}"] = nn.Parameter(torch.eye(num_heads))
            self.proj[f"mlp_{i}"] = nn.Parameter(torch.eye(model0.transformer.layers[i][3].net[0].weight.shape[0]))

        self.model = copy.deepcopy(model0)
        self.model.pos_embedding = EmbeddingMerger(model0.pos_embedding, model1.pos_embedding)
        self.model.to_patch_embedding[2] = LinearMerger(model0.to_patch_embedding[2], model1.to_patch_embedding[2])
        self.model.to_patch_embedding[2].set_P_in(torch.eye(patch_dim, device=device))
        self.model.to_patch_embedding[1] = LayerNormMerger(model0.to_patch_embedding[1], model1.to_patch_embedding[1])
        self.model.to_patch_embedding = PatchEmbeddingMerger(self.model.to_patch_embedding)

        for i in range(n_layers):
            layer = self.model.transformer.layers[i]
            layer[3].net[0] = LinearMerger(model0.transformer.layers[i][3].net[0], model1.transformer.layers[i][3].net[0])
            layer[3].net[3] = LinearMerger(model0.transformer.layers[i][3].net[3], model1.transformer.layers[i][3].net[3])
            layer[1].to_qkv = AttnQKVMerger(model0.transformer.layers[i][1].to_qkv, model1.transformer.layers[i][1].to_qkv, num_heads, embed_dim)
            layer[1].to_out[0] = AttnOutMerger(model0.transformer.layers[i][1].to_out[0], model1.transformer.layers[i][1].to_out[0], num_heads, embed_dim)
            layer[0] = RMSMerger(model0.transformer.layers[i][0], model1.transformer.layers[i][0], embed_dim)
            layer[2] = RMSMerger(model0.transformer.layers[i][2], model1.transformer.layers[i][2], embed_dim)
            layer[3].net = FeedForwardMerger(layer[3].net)

        self.model.final_norm = RMSMerger(model0.final_norm, model1.final_norm, embed_dim)
        self.model.linear_head = LinearMerger(model0.linear_head, model1.linear_head)
        self.model.linear_head.set_P_out(torch.eye(model0.linear_head.weight.shape[0], device=device))

    def _project(self):
        if self.permutations_only:
            P_res = project(self.proj["residual"], "perm")
        else:
            Q, _R = torch.linalg.qr(self.proj["residual"])
            P_res = Q
        P_res_t = P_res.t()
        self.model.pos_embedding.set_P(P_res)
        self.model.to_patch_embedding[2].set_P_out(P_res)
        for i in range(len(self.model.transformer.layers)):
            layer = self.model.transformer.layers[i]
            P_mlp = project(self.proj[f"mlp_{i}"], "perm")
            layer[3].net[0].set_P_in(P_res_t);  layer[3].net[0].set_P_out(P_mlp)
            layer[3].net[3].set_P_in(P_mlp.t()); layer[3].net[3].set_P_out(P_res)
            P_heads = project(self.proj[f"attention_heads_{i}"], "perm")
            layer[1].to_qkv.set_P_in(P_res_t);   layer[1].to_qkv.set_P_out(P_heads)
            layer[1].to_out[0].set_P_in(P_heads); layer[1].to_out[0].set_P_out(P_res)
            layer[0].set_P(P_res_t); layer[2].set_P(P_res_t)
        self.model.final_norm.set_P(P_res_t)
        self.model.linear_head.set_P_in(P_res_t)

    def forward(self, img, coeff):
        self._project()
        return self.model(img, coeff=coeff)


@torch.no_grad()
def _corr_perm(X0, X1):
    X0 = X0.float(); X1 = X1.float()
    X0 = (X0 - X0.mean(0)) / (X0.std(0) + 1e-8)
    X1 = (X1 - X1.mean(0)) / (X1.std(0) + 1e-8)
    C = -(X0.T @ X1) / max(X0.shape[0], 1)
    return otify(C.cpu()).to(X0.device)

def _mlp_preact(model, x, layer_i):
    x = model.to_patch_embedding(x)
    b, n, _ = x.shape
    x = x + model.pos_embedding(torch.arange(n, device=x.device).expand(b, -1))
    for i, (norm_attn, attn, norm_ff, ff) in enumerate(model.transformer.layers):
        x = attn(norm_attn(x)) + x
        h = norm_ff(x)
        if i == layer_i:
            return ff.net[0](h).reshape(-1, ff.net[0].out_features)
        x = ff(h) + x
    raise IndexError(layer_i)

def _head_acts(model, x, layer_i, heads, dim):
    x = model.to_patch_embedding(x)
    b, n, _ = x.shape
    x = x + model.pos_embedding(torch.arange(n, device=x.device).expand(b, -1))
    for i, (norm_attn, attn, norm_ff, ff) in enumerate(model.transformer.layers):
        h = norm_attn(x)
        if i == layer_i:
            qkv = attn.to_qkv(h).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), qkv)
            a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * attn.scale, dim=-1)
            out = torch.matmul(a, v)
            return out.abs().mean(dim=(2, 3)).reshape(-1, heads)
        x = attn(h) + x
        x = ff(norm_ff(x)) + x
    raise IndexError(layer_i)

@torch.no_grad()
def activation_matching(model0, model1, heads, loader, device, max_batches=20, iterations=3):
    m0 = model0.eval().to(device)
    m1 = copy.deepcopy(model1).eval().to(device)
    dim = m0.pos_embedding.weight.shape[1]
    n_layers = len(m0.transformer.layers)
    for it in range(iterations):
        mlp_acts = {i: ([], []) for i in range(n_layers)}
        head_acts = {i: ([], []) for i in range(n_layers)}
        res0, res1 = [], []
        for bi, (imgs, _) in enumerate(loader):
            if bi >= max_batches:
                break
            imgs = imgs.to(device)

            def _resid(m, imgs):
                x = m.to_patch_embedding(imgs)
                b, n, _ = x.shape
                x = x + m.pos_embedding(torch.arange(n, device=x.device).expand(b, -1))
                x = m.transformer(x)
                return x.mean(1)

            res0.append(_resid(m0, imgs))
            res1.append(_resid(m1, imgs))
            for li in range(n_layers):
                mlp_acts[li][0].append(_mlp_preact(m0, imgs, li))
                mlp_acts[li][1].append(_mlp_preact(m1, imgs, li))
                head_acts[li][0].append(_head_acts(m0, imgs, li, heads, dim))
                head_acts[li][1].append(_head_acts(m1, imgs, li, heads, dim))


        R0 = torch.cat(res0, 0); R1 = torch.cat(res1, 0)
        ortho_residual(m1, _corr_perm(R0, R1))
        for li in range(n_layers):
            A0 = torch.cat(mlp_acts[li][0], 0); A1 = torch.cat(mlp_acts[li][1], 0)
            permute_mlp(m1, _corr_perm(A0, A1), li)
            H0 = torch.cat(head_acts[li][0], 0); H1 = torch.cat(head_acts[li][1], 0)
            permute_heads(m1, _corr_perm(H0, H1), heads, dim, li)
        print(f"[activation_matching] iter {it + 1}/{iterations} done")
    return m1


@torch.no_grad()
def evaluate(model, loader, device, coeff=None, max_batches=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x) if coeff is None else model(x, coeff=coeff)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return 100.0 * correct / total, loss_sum / total

def interpolate_state_dicts(sd0, sd1, coeff):
    return {k: coeff * sd0[k] + (1 - coeff) * sd1[k] for k in sd0}

def summarize(coeff_losses):
    L0 = coeff_losses[1.0]; L1 = coeff_losses[0.0]
    barriers, max_barrier = {}, -float("inf")
    for c, L in coeff_losses.items():
        expected = c * L0 + (1 - c) * L1
        barriers[c] = L - expected
        max_barrier = max(max_barrier, barriers[c])
    return max_barrier, coeff_losses[0.5], barriers

def sweep_state_dict(model_ctor, sd0, sd1, loader, device, coeffs=None, max_batches=None):
    if coeffs is None:
        coeffs = [round(0.1 * i, 1) for i in range(11)]
    model = model_ctor().to(device)
    losses, accs = {}, {}
    for c in coeffs:
        model.load_state_dict(interpolate_state_dicts(sd0, sd1, c))
        acc, loss = evaluate(model, loader, device, coeff=None, max_batches=max_batches)
        losses[c], accs[c] = loss, acc
    return losses, accs

def sweep_merger(merger, loader, device, coeffs=None, max_batches=None):
    if coeffs is None:
        coeffs = [round(0.1 * i, 1) for i in range(11)]
    losses, accs = {}, {}
    for c in coeffs:
        acc, loss = evaluate(merger, loader, device, coeff=c, max_batches=max_batches)
        losses[c], accs[c] = loss, acc
    return losses, accs

def report(name, losses):
    mb, mid, _ = summarize(losses)
    ends = (losses[0.0] + losses[1.0]) / 2
    print(f"  {name}: max_barrier={mb:.4f}  mid_loss={mid:.4f}  ends={ends:.4f}")

print("[lib] model + algorithms ready")


IMG_EXTS = {".jpg", ".jpeg", ".png"}

def resolve_eurosat_root(root):
    """Find the RGB EuroSAT ImageFolder root: the directory whose class
    subfolders contain the most jpg/png images (skips the multispectral
    ``EuroSATallBands`` .tif copy that ships in the Kaggle dataset)."""
    best, best_n = None, 0
    for d in [Path(root), *Path(root).rglob("*")]:
        if not d.is_dir() or "allband" in d.name.lower().replace(" ", ""):
            continue
        subs = [c for c in d.iterdir() if c.is_dir()]
        if len(subs) < 5:
            continue
        n = sum(1 for c in subs for f in c.iterdir()
                if f.suffix.lower() in IMG_EXTS)
        if n > best_n:
            best, best_n = d, n
    if best is None:
        raise FileNotFoundError(f"EuroSAT class folders not found under {root}")
    return best

def _decode_all(root_dir, cache_path):
    """One-time jpg decode of the whole dataset into a uint8 tensor, cached."""
    if cache_path.exists():
        blob = torch.load(cache_path)
        return blob["images"], blob["labels"], blob["classes"]


    ds = torchvision.datasets.ImageFolder(str(root_dir), transform=TF.pil_to_tensor)
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=3)
    imgs, lbls = [], []
    for xb, yb in loader:
        imgs.append(xb)
        lbls.append(yb)
    images, labels = torch.cat(imgs), torch.cat(lbls)
    torch.save({"images": images, "labels": labels, "classes": ds.classes}, cache_path)
    print(f"[data] decoded EuroSAT: {tuple(images.shape)} -> cached {cache_path}")
    return images, labels, ds.classes

def _stratified_split(labels, test_frac, seed):
    """Deterministic per-class 80/20 split (EuroSAT has no official one)."""
    g = torch.Generator().manual_seed(seed)
    train_idx, test_idx = [], []
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        n_test = int(round(len(idx) * test_frac))
        test_idx.append(perm[:n_test])
        train_idx.append(perm[n_test:])
    return torch.cat(train_idx), torch.cat(test_idx)

class GPULoader:
    """DataLoader stand-in over GPU-resident uint8 images. Batching,
    augmentation, and normalization all run on the GPU; yields (x, y)
    already on device (so downstream ``x.to(device)`` is a no-op)."""

    def __init__(self, images, labels, batch_size, shuffle, mean, std,
                 augment=None, drop_last=False):
        self.images, self.labels = images, labels
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.augment = augment
        dev = images.device
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)

    def __len__(self):
        n = self.images.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = self.images.shape[0]
        dev = self.images.device
        order = (torch.randperm(n, device=dev) if self.shuffle
                 else torch.arange(n, device=dev))
        end = n - (n % self.batch_size) if self.drop_last else n
        for s in range(0, end, self.batch_size):
            idx = order[s:s + self.batch_size]
            x = self.images[idx].float() / 255.0
            if self.augment is not None:
                x = self.augment(x)
            yield (x - self.mean) / self.std, self.labels[idx]

def get_eurosat_loaders(root, device, image_size=64, batch_size=128,
                        test_batch=2048, test_frac=0.2, split_seed=42,
                        cache_dir="eurosat_cache"):
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    eurosat_root = resolve_eurosat_root(root)
    images, labels, classes = _decode_all(eurosat_root, cache / "eurosat_all.pt")
    tr_idx, te_idx = _stratified_split(labels, test_frac, split_seed)
    tr_imgs, tr_lbls = images[tr_idx].to(device), labels[tr_idx].to(device)
    te_imgs, te_lbls = images[te_idx].to(device), labels[te_idx].to(device)


    _xf = tr_imgs.float() / 255.0
    mean = tuple(_xf.mean(dim=(0, 2, 3)).tolist())
    std = tuple(_xf.std(dim=(0, 2, 3)).tolist())
    del _xf
    print(f"[data] EuroSAT train stats  mean={[round(m, 4) for m in mean]}  "
          f"std={[round(s, 4) for s in std]}")


    augment = nn.Sequential(
        K.RandomCrop((image_size, image_size), padding=image_size // 8),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJiggle(0.4, 0.4, 0.4, 0.1, p=1.0),
    ).to(device)

    trainloader = GPULoader(tr_imgs, tr_lbls, batch_size, shuffle=True,
                            mean=mean, std=std, augment=augment, drop_last=True)

    statloader = GPULoader(tr_imgs, tr_lbls, batch_size, shuffle=True,
                           mean=mean, std=std)
    testloader = GPULoader(te_imgs, te_lbls, test_batch, shuffle=False,
                           mean=mean, std=std)
    print(f"[data] EuroSAT root={eurosat_root}  train={tr_imgs.shape[0]:,}  "
          f"test={te_imgs.shape[0]:,}  classes={len(classes)}  (all on {device})")
    data_info = {"classes": classes, "mean": mean, "std": std,
                 "test_frac": test_frac, "split_seed": split_seed}
    return trainloader, statloader, testloader, len(classes), data_info


cfg = {
    "dataset": "EuroSAT",

    "image_size": 64,
    "patch_size": 8,
    "dim": 256,
    "depth": 6,
    "heads": 8,
    "mlp_dim": 512,
    "dropout": 0.2,
    "bias": True,

    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-3,
}

SMOKE = False

EPOCHS = 150
MERGER_EPOCHS = 15
WM_ITERS = 15
ACT_BATCHES = 20
SWEEP_MAX_BATCHES = None
TEST_FRAC = 0.2

if SMOKE:
    EPOCHS, MERGER_EPOCHS, WM_ITERS = 1, 1, 2
    ACT_BATCHES, SWEEP_MAX_BATCHES = 2, 3

print(f"[cfg] SMOKE={SMOKE}  epochs={EPOCHS}  merger_epochs={MERGER_EPOCHS} "
      f"wm_iters={WM_ITERS}  sweep_batches={SWEEP_MAX_BATCHES}")


OUT_DIR = Path("outputs_eurosat")
MODELS_DIR = OUT_DIR / "models"
LOGS_DIR = OUT_DIR / "logs"
MERGERS_DIR = OUT_DIR / "mergers"
PLOTS_DIR = OUT_DIR / "plots"
for _d in (MODELS_DIR, LOGS_DIR, MERGERS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
print(f"[out] saving everything under {OUT_DIR.resolve()}")


import kagglehub

raw_path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
print("Path to dataset files:", raw_path)

trainloader, statloader, testloader, num_classes, data_info = get_eurosat_loaders(
    raw_path,
    device,
    image_size=cfg["image_size"],
    batch_size=cfg["batch_size"],
    test_batch=2048,
    test_frac=TEST_FRAC,
)

(LOGS_DIR / "data_info.json").write_text(json.dumps(data_info, indent=2))


def train_one_vit(seed, trainloader, testloader, epochs,
                  quick_batches=None, eval_max_batches=40):
    torch.manual_seed(seed)
    model = build_vit(cfg, num_classes, image_size=cfg["image_size"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = device == "cuda"

    history = []
    log_path = LOGS_DIR / f"vit_seed{seed}_train_log.json"
    ckpt_path = MODELS_DIR / f"vit_seed{seed}.pth"
    resume_path = MODELS_DIR / f"vit_seed{seed}_resume.pth"


    start_epoch = 0
    if log_path.exists() and ckpt_path.exists():
        history = json.loads(log_path.read_text())
        start_epoch = min(len(history), epochs)
        if start_epoch > 0:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            if resume_path.exists():
                optimizer.load_state_dict(
                    torch.load(resume_path, map_location=device)["optimizer"])
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  seed{seed}: resumed at epoch {start_epoch}/{epochs} from {ckpt_path}")

    for epoch in range(start_epoch, epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for i, (x, y) in enumerate(trainloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                logits, _ = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)
            if quick_batches and i + 1 >= quick_batches:
                break
        scheduler.step()
        acc, vloss = evaluate(model, testloader, device, max_batches=eval_max_batches)
        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss / tr_total,
            "train_acc": 100.0 * tr_correct / tr_total,
            "test_acc": acc,
            "test_loss": vloss,
            "lr": scheduler.get_last_lr()[0],
        })

        log_path.write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), ckpt_path)
        torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch + 1}, resume_path)
        print(f"  seed{seed} ep{epoch + 1}/{epochs}: "
              f"train_loss={history[-1]['train_loss']:.4f} "
              f"train_acc={history[-1]['train_acc']:.2f}% "
              f"test_acc={acc:.2f}% test_loss={vloss:.4f}")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  saved {ckpt_path} and {log_path}")
    return model, history


print("=" * 20, "TRAIN ViT seed=0", "=" * 20)
model0, history0 = train_one_vit(0, trainloader, testloader, EPOCHS,
                                 quick_batches=3 if SMOKE else None)


print("=" * 20, "TRAIN ViT seed=1", "=" * 20)
model1, history1 = train_one_vit(1, trainloader, testloader, EPOCHS,
                                 quick_batches=3 if SMOKE else None)


print("### Endpoint metrics (full test set)")
_endpoint = {}
for _nm, _m in (("seed0", model0), ("seed1", model1)):
    _acc, _loss = evaluate(_m, testloader, device)
    _endpoint[_nm] = {"test_acc": _acc, "test_loss": _loss}
    print(f"  {_nm}: test_acc={_acc:.2f}%  test_loss={_loss:.4f}")
(LOGS_DIR / "endpoint_metrics.json").write_text(json.dumps(_endpoint, indent=2))


r0 = reparameterize(model0, num_classes, image_size=cfg["image_size"]).to(device).eval()
r1 = reparameterize(model1, num_classes, image_size=cfg["image_size"]).to(device).eval()
heads = cfg["heads"]
print("[reparam] r0, r1 ready (RMSNorm canonical form)")


curve_van, acc_van = sweep_state_dict(
    lambda: copy.deepcopy(r0), r0.state_dict(), r1.state_dict(),
    testloader, device, max_batches=SWEEP_MAX_BATCHES,
)
print("[eval] Vanilla averaging")
report("Vanilla averaging", curve_van)
(LOGS_DIR / "curve_vanilla.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_van.items()},
     "acc": {str(k): v for k, v in acc_van.items()}}, indent=2))


_a0 = copy.deepcopy(r0)
_a1 = activation_matching(_a0, copy.deepcopy(r1), heads, statloader, device,
                          max_batches=ACT_BATCHES, iterations=3)
torch.save(_a1.state_dict(), MODELS_DIR / "vit_seed1_activation_matched.pth")
curve_act, acc_act = sweep_state_dict(
    lambda: copy.deepcopy(_a0), _a0.state_dict(), _a1.state_dict(),
    testloader, device, max_batches=SWEEP_MAX_BATCHES,
)
print("[eval] Activation matching")
report("Activation matching", curve_act)
(LOGS_DIR / "curve_activation_matching.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_act.items()},
     "acc": {str(k): v for k, v in acc_act.items()}}, indent=2))


_w0, _w1 = copy.deepcopy(r0), copy.deepcopy(r1)
weight_matching(_w0, _w1, heads, iterations=WM_ITERS)
torch.save(_w1.state_dict(), MODELS_DIR / "vit_seed1_weight_matched.pth")
_wm_merger = ViTMerger(_w0, _w1, num_heads=heads, device=device).to(device)
curve_wm, acc_wm = sweep_merger(_wm_merger, testloader, device, max_batches=SWEEP_MAX_BATCHES)
print("[eval] Weight matching (ours)")
report("Weight matching (ours)", curve_wm)
(LOGS_DIR / "curve_weight_matching.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_wm.items()},
     "acc": {str(k): v for k, v in acc_wm.items()}}, indent=2))


def train_learned_merger(r0, r1, heads, trainloader, testloader, *, name,
                         permutations_only, epochs, wm_iters,
                         lr=1e-3, quick_batches=None, eval_max_batches=40, seed=41):
    torch.manual_seed(seed)
    random.seed(seed)
    m0, m1 = copy.deepcopy(r0), copy.deepcopy(r1)
    print(f"[merger] weight matching ({wm_iters} iters, permutations_only={permutations_only})...")
    weight_matching(m0, m1, heads, iterations=wm_iters)

    merger = ViTMerger(m0, m1, num_heads=heads, device=device,
                       permutations_only=permutations_only).to(device)
    optimizer = optim.Adam([p for p in merger.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    sample_coeff = lambda: random.uniform(0.4, 0.6)

    history = []
    log_path = LOGS_DIR / f"merger_{name}_train_log.json"
    ckpt_path = MERGERS_DIR / f"merger_{name}.pth"
    resume_path = MERGERS_DIR / f"merger_{name}_resume.pth"


    start_epoch = 0
    if log_path.exists() and ckpt_path.exists():
        history = json.loads(log_path.read_text())
        start_epoch = min(max(len(history) - 1, 0), epochs)
        if start_epoch > 0:
            merger.load_state_dict(torch.load(ckpt_path, map_location=device))
            if resume_path.exists():
                optimizer.load_state_dict(
                    torch.load(resume_path, map_location=device)["optimizer"])
            print(f"  merger {name}: resumed at epoch {start_epoch}/{epochs} from {ckpt_path}")

    if not history:
        acc, loss = evaluate(merger, testloader, device, coeff=0.5, max_batches=eval_max_batches)
        print(f"  weight-matched midpoint: acc={acc:.2f}% loss={loss:.4f}")
        history.append({"epoch": 0, "midpoint_acc": acc, "midpoint_loss": loss})
    for epoch in range(start_epoch, epochs):
        merger.train()
        tr_loss, tr_total = 0.0, 0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(merger(x, coeff=sample_coeff())[0], y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            if quick_batches and i + 1 >= quick_batches:
                break
        acc, loss = evaluate(merger, testloader, device, coeff=0.5, max_batches=eval_max_batches)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss / tr_total,
                        "midpoint_acc": acc, "midpoint_loss": loss})
        log_path.write_text(json.dumps(history, indent=2))
        torch.save(merger.state_dict(), ckpt_path)
        torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch + 1}, resume_path)
        print(f"  ep{epoch + 1}/{epochs}: midpoint_acc={acc:.2f}% midpoint_loss={loss:.4f}")
    torch.save(merger.state_dict(), ckpt_path)
    print(f"  saved {ckpt_path} and {log_path}")
    return merger


print("=" * 20, "LEARNED MATCHING (OURS)", "=" * 20)
_merger_ortho = train_learned_merger(
    r0, r1, heads, trainloader, testloader, name="ours",
    permutations_only=False, epochs=MERGER_EPOCHS, wm_iters=WM_ITERS,
    quick_batches=3 if SMOKE else None,
)
curve_lm, acc_lm = sweep_merger(_merger_ortho, testloader, device, max_batches=SWEEP_MAX_BATCHES)
print("[eval] Learned matching (ours)")
report("Learned matching (ours)", curve_lm)
(LOGS_DIR / "curve_learned_matching_ours.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_lm.items()},
     "acc": {str(k): v for k, v in acc_lm.items()}}, indent=2))


print("=" * 20, "LEARNED MATCHING (PERMUTATIONS)", "=" * 20)
_merger_perm = train_learned_merger(
    r0, r1, heads, trainloader, testloader, name="permutations",
    permutations_only=True, epochs=MERGER_EPOCHS, wm_iters=WM_ITERS,
    quick_batches=3 if SMOKE else None,
)
curve_lmp, acc_lmp = sweep_merger(_merger_perm, testloader, device, max_batches=SWEEP_MAX_BATCHES)
print("[eval] Learned matching (permutations)")
report("Learned matching (permutations)", curve_lmp)
(LOGS_DIR / "curve_learned_matching_permutations.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_lmp.items()},
     "acc": {str(k): v for k, v in acc_lmp.items()}}, indent=2))


curves = {
    "Vanilla averaging": curve_van,
    "Activation matching": curve_act,
    "Weight matching (ours)": curve_wm,
    "Learned matching (permutations)": curve_lmp,
    "Learned matching (ours)": curve_lm,
}
acc_curves = {
    "Vanilla averaging": acc_van,
    "Activation matching": acc_act,
    "Weight matching (ours)": acc_wm,
    "Learned matching (permutations)": acc_lmp,
    "Learned matching (ours)": acc_lm,
}
style = {
    "Vanilla averaging": dict(color="#2ca02c", marker="D"),
    "Activation matching": dict(color="#9467bd", marker="s"),
    "Weight matching (ours)": dict(color="#ff7f0e", marker="v"),
    "Learned matching (permutations)": dict(color="#d62728", marker="^"),
    "Learned matching (ours)": dict(color="#1f77b4", marker="o"),
}

def _lambda_plot(data, ylabel, fname):
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for name, series in data.items():
        st = style[name]
        xs = sorted(series)
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8, label=name)
    ax.set_xlabel(r"Interpolation coefficient ($\lambda$)")
    ax.set_ylabel(ylabel)
    ax.set_title("EuroSAT · Linear Mode Connectivity of two ViTs")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.subplots_adjust(right=0.62)
    fig.savefig(PLOTS_DIR / fname, dpi=200, bbox_inches="tight", facecolor="white")
    return fig

fig_loss = _lambda_plot(curves, "Test loss", "eurosat_loss_vs_lambda.png")
fig_acc = _lambda_plot(acc_curves, "Test accuracy (%)", "eurosat_acc_vs_lambda.png")


fig_tr, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
for _sd, _hist, _col in (("seed0", history0, "#1f77b4"), ("seed1", history1, "#d62728")):
    _eps = [h["epoch"] for h in _hist]
    ax_l.plot(_eps, [h["train_loss"] for h in _hist], color=_col, linestyle="--",
              label=f"{_sd} train")
    ax_l.plot(_eps, [h["test_loss"] for h in _hist], color=_col, label=f"{_sd} test")
    ax_a.plot(_eps, [h["train_acc"] for h in _hist], color=_col, linestyle="--",
              label=f"{_sd} train")
    ax_a.plot(_eps, [h["test_acc"] for h in _hist], color=_col, label=f"{_sd} test")
ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss"); ax_l.legend(fontsize=8); ax_l.grid(True)
ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy (%)"); ax_a.legend(fontsize=8); ax_a.grid(True)
fig_tr.suptitle("EuroSAT · Base ViT training (seeds 0 & 1)")
fig_tr.tight_layout()
fig_tr.savefig(PLOTS_DIR / "eurosat_training_curves.png", dpi=200,
               bbox_inches="tight", facecolor="white")

barriers = {k: float(summarize(v)[0]) for k, v in curves.items()}
serial = {
    "loss_curves": {k: {str(c): float(v) for c, v in d.items()} for k, d in curves.items()},
    "acc_curves": {k: {str(c): float(v) for c, v in d.items()} for k, d in acc_curves.items()},
    "barriers": barriers,
}
(OUT_DIR / "eurosat_interp_curves.json").write_text(json.dumps(serial, indent=2))
print(f"[save] plots -> {PLOTS_DIR}  curves+barriers -> {OUT_DIR / 'eurosat_interp_curves.json'}")
print("[barriers]")
for k, v in sorted(barriers.items(), key=lambda kv: kv[1]):
    print(f"  {k:34s} max_barrier={v:.4f}")


import zipfile

def _zip_results():
    zip_path = Path("eurosat_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(Path("outputs_eurosat").rglob("*")):
            if f.is_file():
                zf.write(f, f.as_posix())
    print(f"[zip] {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path.read_bytes()

results_download
