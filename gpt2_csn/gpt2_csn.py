"""Linear mode connectivity of two GPT-2 models on CodeSearchNet-Python (GLMC)."""

import copy
import gzip
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from scipy.optimize import linear_sum_assignment
import ot
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


device = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16

if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu] {_gpu_name}  ({_vram:.0f} GB VRAM)  device={device}")
else:
    print("[gpu] CUDA not available — running on CPU (will be very slow)")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, bias=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        norm_x = x / rms
        out = self.weight * norm_x
        if self.bias is not None:
            out = out + self.bias
        return out

def absorb_ln_scale(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.ln_1.bias.copy_(block.ln_1.bias / block.ln_1.weight)
            block.attn.c_attn.weight.copy_(block.attn.c_attn.weight * block.ln_1.weight.unsqueeze(1))
            block.ln_1.weight.copy_(torch.ones(block.ln_1.weight.shape))

            block.ln_2.bias.copy_(block.ln_2.bias / block.ln_2.weight)
            block.mlp.c_fc.weight.copy_(block.mlp.c_fc.weight * block.ln_2.weight.unsqueeze(1))
            block.ln_2.weight.copy_(torch.ones(block.ln_2.weight.shape))

        with torch.no_grad():
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
        model.transformer.ln_f.bias.copy_(model.transformer.ln_f.bias / model.transformer.ln_f.weight)
        model.lm_head.weight.copy_(model.lm_head.weight * model.transformer.ln_f.weight)
        model.transformer.ln_f.weight.copy_(torch.ones(model.transformer.ln_f.weight.shape))

def replace_layernorm(module):
    with torch.no_grad():
        for name, child in module.named_children():
            replace_layernorm(child)
            if isinstance(child, nn.LayerNorm):
                rms_norm = RMSNorm(child.normalized_shape, eps=child.eps, bias=True)
                nn.init.ones_(rms_norm.weight)
                rms_norm.bias.copy_(child.bias)
                setattr(module, name, rms_norm)

def apply_mean_subtraction_to_weights(model):
    dim = model.transformer.h[0].ln_1.bias.shape[0]
    M = torch.eye(dim) - torch.ones(dim, dim) / dim
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.copy_(block.attn.c_proj.weight @ M)
            block.attn.c_proj.bias.copy_(block.attn.c_proj.bias @ M)
            block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight @ M)
            block.mlp.c_proj.bias.copy_(block.mlp.c_proj.bias @ M)
        model.transformer.wte.weight.copy_(model.transformer.wte.weight @ M)
        model.transformer.wpe.weight.copy_(model.transformer.wpe.weight @ M)

def permute_mlp(model, idx, P):
    with torch.no_grad():
        model.transformer.h[idx].mlp.c_fc.weight.copy_(model.transformer.h[idx].mlp.c_fc.weight @ P)
        model.transformer.h[idx].mlp.c_fc.bias.copy_(model.transformer.h[idx].mlp.c_fc.bias @ P)
        model.transformer.h[idx].mlp.c_proj.weight.copy_(P.t() @ model.transformer.h[idx].mlp.c_proj.weight)

def permute_heads(model, layer_idx, P):
    with torch.no_grad():
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])

        attn = copy.deepcopy(model.transformer.h[layer_idx].attn)
        num_heads = attn.num_heads
        embed_dim = attn.embed_dim
        c_attn = attn.c_attn
        c_proj = attn.c_proj

        c_attn_weight = torch.cat((c_attn.weight.t(), c_attn.bias.data.t().unsqueeze(1)), dim=1)
        Q, K, V = c_attn_weight.data.chunk(3, dim=0)
        Q = rearrange(Q, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
        K = rearrange(K, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
        V = rearrange(V, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)

        Q = permute(Q, P); K = permute(K, P); V = permute(V, P)

        OUT = rearrange(c_proj.weight.data.t(), ' m (h d) -> m h d', h=num_heads, m=embed_dim)
        OUT = OUT.permute(1, 2, 0)
        OUT = permute(OUT, P)

        QK = torch.bmm(Q.transpose(1, 2), K)
        OUTV = OUT.transpose(1, 2) @ V

        Q_new = torch.zeros(QK.shape); K_new = torch.zeros(QK.shape); V_new = torch.zeros(QK.shape)
        OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1])

        for head_i in range(QK.size(0)):
            def split(A):
                return A, torch.eye(A.shape[1])
            U_S_r, V_r = split(QK[head_i])
            Q_new[head_i] = U_S_r.t()
            K_new[head_i] = V_r
            U_S_r, V_r = split(OUTV[head_i])
            OUT_new[head_i] = U_S_r.t()
            V_new[head_i] = V_r

        Q_new = Q_new.reshape(-1, embed_dim + 1)
        K_new = K_new.reshape(-1, embed_dim + 1)
        V_new = V_new.reshape(-1, embed_dim + 1)

        c_attn.weight.data = torch.cat((Q_new, K_new, V_new), dim=0)[:, :-1].t()
        c_attn.bias.data = torch.cat((Q_new, K_new, V_new), dim=0)[:, -1:].t()

        OUT_new = OUT_new.permute(2, 0, 1).reshape(embed_dim, -1).t()
        c_proj.weight.data = OUT_new

        model.transformer.h[layer_idx].attn.c_attn.nx = c_attn.weight.shape[0]
        model.transformer.h[layer_idx].attn.c_attn.nf = c_attn.weight.shape[1]
        model.transformer.h[layer_idx].attn.c_attn.weight = torch.nn.Parameter(c_attn.weight.clone())
        model.transformer.h[layer_idx].attn.c_attn.bias = torch.nn.Parameter(c_attn.bias.clone().squeeze())
        model.transformer.h[layer_idx].attn.c_proj.nx = c_proj.weight.shape[0]
        model.transformer.h[layer_idx].attn.c_proj.nf = c_proj.weight.shape[1]
        model.transformer.h[layer_idx].attn.c_proj.weight = torch.nn.Parameter(c_proj.weight.clone())
        model.transformer.h[layer_idx].attn.split_size = c_attn.weight.shape[1] // 3
        model.transformer.h[layer_idx].attn.head_dim = model.transformer.h[layer_idx].attn.embed_dim + 1

def project_to_attn_circuits(model, layer_idx):
    with torch.no_grad():
        attn = copy.deepcopy(model.transformer.h[layer_idx].attn)
        num_heads = attn.num_heads
        embed_dim = attn.embed_dim
        c_attn = attn.c_attn
        c_proj = attn.c_proj

        c_attn_weight = torch.cat((c_attn.weight.t(), c_attn.bias.data.t().unsqueeze(1)), dim=1)
        Q, K, V = c_attn_weight.data.chunk(3, dim=0)
        Q = rearrange(Q, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
        K = rearrange(K, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
        V = rearrange(V, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
        OUT = rearrange(c_proj.weight.data.t(), ' m (h d) -> m h d', h=num_heads, m=embed_dim)
        OUT = OUT.permute(1, 2, 0)

        QK = torch.bmm(Q.transpose(1, 2), K)
        OUTV = OUT.transpose(1, 2) @ V

        Q_new = torch.zeros(QK.shape); K_new = torch.zeros(QK.shape); V_new = torch.zeros(QK.shape)
        OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1])

        for head_i in range(QK.size(0)):
            def split(A):
                return A, torch.eye(A.shape[1])
            U_S_r, V_r = split(QK[head_i])
            Q_new[head_i] = U_S_r.t() * (U_S_r.shape[1] ** 0.5 / V.shape[1] ** 0.5)
            K_new[head_i] = V_r
            U_S_r, V_r = split(OUTV[head_i])
            OUT_new[head_i] = U_S_r.t()
            V_new[head_i] = V_r

        Q_new = Q_new.reshape(-1, embed_dim + 1)
        K_new = K_new.reshape(-1, embed_dim + 1)
        V_new = V_new.reshape(-1, embed_dim + 1)

        c_attn.weight.data = torch.cat((Q_new, K_new, V_new), dim=0)[:, :-1].t()
        c_attn.bias.data = torch.cat((Q_new, K_new, V_new), dim=0)[:, -1:].t()

        OUT_new = OUT_new.permute(2, 0, 1).reshape(embed_dim, -1).t()
        c_proj.weight.data = OUT_new

        model.transformer.h[layer_idx].attn.c_attn.nx = c_attn.weight.shape[0]
        model.transformer.h[layer_idx].attn.c_attn.nf = c_attn.weight.shape[1]
        model.transformer.h[layer_idx].attn.c_attn.weight = torch.nn.Parameter(c_attn.weight.clone())
        model.transformer.h[layer_idx].attn.c_attn.bias = torch.nn.Parameter(c_attn.bias.clone().squeeze())
        model.transformer.h[layer_idx].attn.c_proj.nx = c_proj.weight.shape[0]
        model.transformer.h[layer_idx].attn.c_proj.nf = c_proj.weight.shape[1]
        model.transformer.h[layer_idx].attn.c_proj.weight = torch.nn.Parameter(c_proj.weight.clone())
        model.transformer.h[layer_idx].attn.split_size = c_attn.weight.shape[1] // 3
        model.transformer.h[layer_idx].attn.head_dim = model.transformer.h[layer_idx].attn.embed_dim + 1

def _make_orthogonal(A):
    dev0 = A.device
    A = A.to("cpu")
    U, _, Vt = torch.linalg.svd(A)
    A = A.to(dev0)
    return torch.mm(U.to(dev0), Vt.to(dev0))

def _make_permutation(P):
    row_ind, col_ind = linear_sum_assignment(-P.detach().cpu().numpy())
    P = P * 0
    P[row_ind, col_ind] = 1
    return P

def sinkhorn(A, iters=20, eps=1e-8):
    """Soft permutation (paper App. D.3): exp(A) projected toward the
    Birkhoff polytope by K Sinkhorn-Knopp iterations. Differentiable."""
    Q = torch.exp(A - A.max())
    for _ in range(iters):
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        Q = Q / (Q.sum(dim=0, keepdim=True) + eps)
    return Q

def project(A, matrix_type):
    """'perm' (straight-through Hungarian), 'soft_perm' (Sinkhorn,
    differentiable), or 'ortho' (SVD)."""
    if matrix_type == "perm":
        return _make_permutation(A).detach() + (A - A.detach())
    elif matrix_type == "soft_perm":
        return sinkhorn(A)
    elif matrix_type == "ortho":
        return _make_orthogonal(A)
    raise ValueError(f"Unknown matrix type: {matrix_type}")

def interpolate(W0, W1, coeff):
    return coeff * W0 + (1 - coeff) * W1


def ortho_residual(model, O):
    with torch.no_grad():
        model.transformer.wte.weight.copy_(model.transformer.wte.weight @ O)
        model.transformer.wpe.weight.copy_(model.transformer.wpe.weight @ O)
        for block in model.transformer.h:
            block.ln_1.bias.copy_(O.t() @ block.ln_1.bias)
            block.attn.c_attn.weight.copy_(O.t() @ block.attn.c_attn.weight)
            block.attn.c_proj.weight.copy_(block.attn.c_proj.weight @ O)
            block.attn.c_proj.bias.copy_(block.attn.c_proj.bias @ O)
            block.ln_2.bias.copy_(O.t() @ block.ln_2.bias)
            block.mlp.c_fc.weight.copy_(O.t() @ block.mlp.c_fc.weight)
            block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight @ O)
            block.mlp.c_proj.bias.copy_(block.mlp.c_proj.bias @ O)
        model.transformer.ln_f.bias.copy_(O.t() @ model.transformer.ln_f.bias)
        model.lm_head.weight.copy_(model.lm_head.weight @ O)

def compute_optimal_orthogonal_matrix(t1, t2):
    C = t2.T @ t1
    U, _, Vh = torch.linalg.svd(C)
    return U @ Vh

def get_cost_heads(t0, t1, heads):
    cost_matrix = torch.zeros((heads, heads))
    for i in range(heads):
        for j in range(heads):
            diff = t0[i] - t1[j]
            cost_matrix[i, j] = torch.sqrt(torch.sum(diff ** 2))
    return cost_matrix

def otify(cost):
    ot_map = ot.emd(torch.ones(cost.shape[0]) / cost.shape[0],
                    torch.ones(cost.shape[0]) / cost.shape[0], cost)
    return ot_map * cost.shape[0]

def _ot_cost_matrix(X, Y, metric="euclidean2", eps=1e-8):
    if metric == "euclidean2":
        X2 = (X ** 2).sum(dim=0, keepdim=True)
        Y2 = (Y ** 2).sum(dim=0, keepdim=True)
        C = X2.T + Y2 - 2 * (X.T @ Y)
        return torch.clamp(C, min=0)
    elif metric == "cosine":
        Xn = X / (X.norm(dim=0, keepdim=True) + eps)
        Yn = Y / (Y.norm(dim=0, keepdim=True) + eps)
        return 1.0 - Xn.T @ Yn
    raise ValueError("metric must be 'euclidean2' or 'cosine'")

def compute_optimal_permutation_matrix_ot(t1, t2, metric="euclidean2"):
    assert t1.shape == t2.shape and t1.dim() == 2
    N, M = t1.shape
    C = _ot_cost_matrix(t1, t2, metric=metric).detach().cpu().numpy()
    a = ot.unif(M); b = ot.unif(M)
    T = ot.emd(a, b, C)
    P = torch.from_numpy(T).to(t1.device, dtype=t1.dtype)
    P = (P * M).round()
    return P

def weight_matching(model0, model1, heads, iterations=15, permutations_only=False,
                    token_freqs=None, block_size=None):
    device0 = next(model0.parameters()).device
    active_token_ids = None
    if token_freqs is not None:
        active_token_ids = (token_freqs > 0).nonzero(as_tuple=False).flatten().to(device0)

    for i in range(iterations):
        tok0 = model0.transformer.wte.weight.data
        tok1 = model1.transformer.wte.weight.data
        if active_token_ids is not None:
            tok0 = tok0.index_select(0, active_token_ids)
            tok1 = tok1.index_select(0, active_token_ids)
        pos0 = model0.transformer.wpe.weight.data
        pos1 = model1.transformer.wpe.weight.data
        if block_size is not None:
            pos0 = pos0[:block_size]; pos1 = pos1[:block_size]
        head0 = model0.lm_head.weight.data
        head1 = model1.lm_head.weight.data
        if active_token_ids is not None:
            head0 = head0.index_select(0, active_token_ids)
            head1 = head1.index_select(0, active_token_ids)

        layers_0 = [tok0.t(), pos0.t(), head0.t()]
        layers_1 = [tok1.t(), pos1.t(), head1.t()]
        if i > 0:
            for layer_i, _ in enumerate(model1.transformer.h):
                layers_0.append(model0.transformer.h[layer_i].attn.c_attn.weight.data)
                layers_1.append(model1.transformer.h[layer_i].attn.c_attn.weight.data)
                layers_0.append(model0.transformer.h[layer_i].attn.c_proj.weight.data.t())
                layers_1.append(model1.transformer.h[layer_i].attn.c_proj.weight.data.t())
                layers_0.append(model0.transformer.h[layer_i].mlp.c_fc.weight.data)
                layers_1.append(model1.transformer.h[layer_i].mlp.c_fc.weight.data)
                layers_0.append(model0.transformer.h[layer_i].mlp.c_proj.weight.data.t())
                layers_1.append(model1.transformer.h[layer_i].mlp.c_proj.weight.data.t())

        layers_0 = [layer / layer.shape[1] ** 0.5 for layer in layers_0]
        layers_1 = [layer / layer.shape[1] ** 0.5 for layer in layers_1]

        if permutations_only:
            O = compute_optimal_permutation_matrix_ot(
                torch.cat(layers_0, dim=1).t(), torch.cat(layers_1, dim=1).t())
            O = O.t()
        else:
            O = compute_optimal_orthogonal_matrix(
                torch.cat(layers_0, dim=1).t(), torch.cat(layers_1, dim=1).t())
        ortho_residual(model1, O)

        for layer_i in range(len(model1.transformer.h)):
            def get_qkv(model):
                attn = copy.deepcopy(model.transformer.h[layer_i].attn)
                num_heads = attn.num_heads
                embed_dim = attn.embed_dim
                c_attn_weight = torch.cat((attn.c_attn.weight.t(),
                                           attn.c_attn.bias.data.t().unsqueeze(1)), dim=1)
                Q, K, V = c_attn_weight.data.chunk(3, dim=0)
                Q = rearrange(Q, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
                K = rearrange(K, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
                V = rearrange(V, '(h d) m -> h d m', h=num_heads, m=embed_dim + 1)
                OUT = rearrange(attn.c_proj.weight.data.t(), ' m (h d) -> m h d',
                                h=num_heads, m=embed_dim)
                OUT = OUT.permute(1, 2, 0)
                return torch.bmm(Q.transpose(1, 2), K), OUT.transpose(1, 2) @ V

            QK0, OUTV0 = get_qkv(model0)
            QK1, OUTV1 = get_qkv(model1)
            cost = get_cost_heads(QK0, QK1, heads=heads) + get_cost_heads(OUTV0, OUTV1, heads=heads)
            P = otify(cost).to(QK0.device)
            permute_heads(model1, layer_i, P)

            ff0 = torch.cat((model0.transformer.h[layer_i].mlp.c_fc.weight.data.t(),
                             model0.transformer.h[layer_i].mlp.c_fc.bias.unsqueeze(1),
                             model0.transformer.h[layer_i].mlp.c_proj.weight.data), dim=1)
            ff1 = torch.cat((model1.transformer.h[layer_i].mlp.c_fc.weight.data.t(),
                             model1.transformer.h[layer_i].mlp.c_fc.bias.unsqueeze(1),
                             model1.transformer.h[layer_i].mlp.c_proj.weight.data), dim=1)
            cost_ff = torch.cdist(ff0 / torch.norm(ff0, dim=-1, keepdim=True),
                                  ff1 / torch.norm(ff1, dim=-1, keepdim=True), p=1).cpu()
            P_ff = otify(cost_ff).to(ff0.device)
            permute_mlp(model1, layer_i, P=P_ff.t())
        print(f"[weight_matching] iter {i + 1}/{iterations} done")
    return model1


class Conv1DMerger(nn.Module):
    def __init__(self, conv1d_0, conv1d_1):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.clone().contiguous())
        self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.clone().contiguous())
        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.clone().contiguous())
        self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.clone().contiguous())
        self.P_in = None; self.P_out = None
        self.nf = conv1d_0.nf
        self.coeff = None

    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P
    def set_coeff(self, coeff): self.coeff = coeff

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.conv1d_0_weight, self.P_in @ self.conv1d_1_weight @ self.P_out, self.coeff)
        bias = interpolate(self.conv1d_0_bias, self.conv1d_1_bias @ self.P_out, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        return x.view(size_out)

class LinearMerger(nn.Module):
    """lm_head merger (nn.Linear weights, transposed vs Conv1D)."""
    def __init__(self, linear_0, linear_1):
        super().__init__()
        self.register_buffer("conv1d_0_weight", linear_0.weight.data.t().clone().contiguous())
        self.register_buffer("conv1d_1_weight", linear_1.weight.data.t().clone().contiguous())
        self.P_in = None
        self.nf = self.conv1d_0_weight.shape[1]
        self.coeff = None

    def set_P_in(self, P): self.P_in = P
    def set_coeff(self, coeff): self.coeff = coeff

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.conv1d_0_weight, self.P_in @ self.conv1d_1_weight, self.coeff)
        x = torch.matmul(x.view(-1, x.size(-1)), weight)
        return x.view(size_out)

class Conv1DMergerCATTN(nn.Module):
    def __init__(self, conv1d_0, conv1d_1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.clone().contiguous())
        self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.clone().contiguous())
        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.clone().contiguous())
        self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.clone().contiguous())
        self.P_in = None; self.P_out = None
        self.nf = conv1d_0.nf
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coeff = None

    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P
    def set_coeff(self, coeff): self.coeff = coeff

    def _permute_heads(self, weight, bias, P):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        c_attn_weight = torch.cat((weight.t(), bias.data.t().unsqueeze(1)), dim=1)
        Q, K, V = c_attn_weight.data.chunk(3, dim=0)
        Q = torch.cat((Q[:, :-1] @ self.P_in.t(), Q[:, -1].unsqueeze(1)), dim=-1)
        Q = rearrange(Q, '(h d) m -> h d m', h=self.num_heads, m=self.embed_dim + 1)
        K = rearrange(K, '(h d) m -> h d m', h=self.num_heads, m=self.embed_dim + 1)
        V = rearrange(V, '(h d) m -> h d m', h=self.num_heads, m=self.embed_dim + 1)
        Q = torch.cat((torch.bmm(Q.transpose(1, 2)[:, :, :-1],
                                 self.P_in.t().expand(self.num_heads, -1, -1)),
                       Q.transpose(1, 2)[:, :, -1:]), dim=-1).transpose(1, 2)
        Q = permute(Q, P); K = permute(K, P); V = permute(V, P)
        Q = Q.reshape(-1, self.embed_dim + 1)
        K = K.reshape(-1, self.embed_dim + 1)
        V = V.reshape(-1, self.embed_dim + 1)
        return (torch.cat((Q, K, V), dim=0)[:, :-1].t(),
                torch.cat((Q, K, V), dim=0)[:, -1:].t())

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        w1, b1 = self._permute_heads(self.conv1d_1_weight, self.conv1d_1_bias, self.P_out)
        bias = interpolate(self.conv1d_0_bias, b1, self.coeff)
        weight = interpolate(self.conv1d_0_weight, w1, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        return x.view(size_out)

class Conv1DMergerCPROJ(nn.Module):
    def __init__(self, conv1d_0, conv1d_1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.clone().contiguous())
        self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.clone().contiguous())
        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.clone().contiguous())
        self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.clone().contiguous())
        self.P_in = None; self.P_out = None
        self.nf = conv1d_0.nf
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coeff = None

    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P
    def set_coeff(self, coeff): self.coeff = coeff

    def _permute_heads(self, weight, bias, P):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        OUT = rearrange(weight.t(), ' m (h d) -> m h d', h=self.num_heads, m=self.embed_dim)
        OUT = OUT.permute(1, 2, 0)
        OUT = torch.cat((OUT.transpose(1, 2)[:, :, :-1] @ self.P_out.expand(self.num_heads, -1, -1),
                         OUT.transpose(1, 2)[:, :, -1:]), dim=-1).transpose(1, 2)
        OUT = permute(OUT, P)
        OUT = OUT.permute(2, 0, 1).reshape(self.embed_dim, -1).t()
        return OUT, bias

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        w1, b1 = self._permute_heads(self.conv1d_1_weight @ self.P_out,
                                     self.conv1d_1_bias @ self.P_out, self.P_in)
        bias = interpolate(self.conv1d_0_bias, b1, self.coeff)
        weight = interpolate(self.conv1d_0_weight, w1, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        return x.view(size_out)

class RMSMerger(nn.Module):
    def __init__(self, rmsnorm_0, rmsnorm_1):
        super().__init__()
        self.register_buffer("bias_0", rmsnorm_0.bias.data.clone().contiguous())
        self.register_buffer("bias_1", rmsnorm_1.bias.data.clone().contiguous())
        self.norm = RMSNorm(dim=rmsnorm_0.weight.shape[0], eps=rmsnorm_0.eps, bias=False)
        self.norm.weight = nn.Parameter(torch.ones(rmsnorm_0.weight.shape[0]))
        self.P = None
        for param in self.norm.parameters():
            param.requires_grad = False
        self.coeff = None

    def set_P(self, P): self.P = P
    def set_coeff(self, coeff): self.coeff = coeff

    def forward(self, x):
        x = self.norm(x)
        return x + interpolate(self.bias_0, self.P @ self.bias_1, coeff=self.coeff)

class EmbeddingMerger(nn.Module):
    def __init__(self, embedding_0, embedding_1):
        super().__init__()
        self.embedding_0 = copy.deepcopy(embedding_0)
        self.embedding_1 = copy.deepcopy(embedding_1)
        for param in self.embedding_0.parameters():
            param.requires_grad = False
        for param in self.embedding_1.parameters():
            param.requires_grad = False
        self.P = None
        self.coeff = None

    def set_P(self, P): self.P = P
    def set_coeff(self, coeff): self.coeff = coeff

    def forward(self, x):
        return interpolate(self.embedding_0(x), self.embedding_1(x) @ self.P, coeff=self.coeff)

def canonicalize(model):
    """Official _absorb + attention-circuit projection, with the official
    function-preservation asserts. Model must be on CPU, fp32."""


    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model.config.attn_implementation = "eager"
    model.config._attn_implementation = "eager"
    dummy_input = torch.randint(0, model.lm_head.weight.shape[0], (2, 32))
    outputs_init = model(input_ids=dummy_input).logits
    absorb_ln_scale(model)
    replace_layernorm(model)
    apply_mean_subtraction_to_weights(model)
    outputs_mid = model(input_ids=dummy_input).logits
    assert torch.allclose(outputs_init, outputs_mid, atol=1e-4), "absorb changed outputs!"
    for i in range(len(model.transformer.h)):
        project_to_attn_circuits(model, i)
    outputs_final = model(input_ids=dummy_input).logits
    assert torch.allclose(outputs_init, outputs_final, atol=1e-4), "circuits changed outputs!"
    return model

class GPTMerger(nn.Module):
    def __init__(self, model0, model1, token_freqs=None, permutations_only=False,
                 iterations=15, soft_perm=False):
        super().__init__()
        assert not soft_perm or permutations_only, "soft_perm is for the permutations-only variant"
        model0 = model0.eval(); model1 = model1.eval()
        canonicalize(model0)
        canonicalize(model1)
        self._permutations_only = permutations_only
        self._soft_perm = soft_perm

        def random_parameter(dim0, dim1=None):
            dim1 = dim0 if dim1 is None else dim1


            scale = 4.6 if soft_perm else 1.0
            eye = torch.eye(dim0, dim1)
            noise = torch.randn_like(eye) * 1e-2
            return nn.Parameter(scale * eye + noise)

        embed_dim = model0.transformer.wte.weight.shape[1]
        num_heads = model0.transformer.h[0].attn.num_heads

        weight_matching(model0, model1, heads=num_heads, iterations=iterations,
                        token_freqs=token_freqs, permutations_only=permutations_only)

        self.proj = nn.ParameterDict({"residual": random_parameter(embed_dim)})
        for i in range(len(model0.transformer.h)):
            self.proj[f"attention_heads_{i}"] = random_parameter(num_heads)
            self.proj[f"mlp_{i}"] = random_parameter(model0.transformer.h[i].mlp.c_fc.bias.shape[0])

        self.model = copy.deepcopy(model0)
        self.model.transformer.wte = EmbeddingMerger(model0.transformer.wte, model1.transformer.wte)
        self.model.transformer.wpe = EmbeddingMerger(model0.transformer.wpe, model1.transformer.wpe)
        for i in range(len(self.model.transformer.h)):
            self.model.transformer.h[i].ln_1 = RMSMerger(model0.transformer.h[i].ln_1, model1.transformer.h[i].ln_1)
            self.model.transformer.h[i].attn.c_attn = Conv1DMergerCATTN(
                model0.transformer.h[i].attn.c_attn, model1.transformer.h[i].attn.c_attn,
                num_heads=num_heads, embed_dim=embed_dim)
            self.model.transformer.h[i].attn.c_proj = Conv1DMergerCPROJ(
                model0.transformer.h[i].attn.c_proj, model1.transformer.h[i].attn.c_proj,
                num_heads=num_heads, embed_dim=embed_dim)
            self.model.transformer.h[i].mlp.c_fc = Conv1DMerger(
                model0.transformer.h[i].mlp.c_fc, model1.transformer.h[i].mlp.c_fc)
            self.model.transformer.h[i].mlp.c_proj = Conv1DMerger(
                model0.transformer.h[i].mlp.c_proj, model1.transformer.h[i].mlp.c_proj)
            self.model.transformer.h[i].ln_2 = RMSMerger(model0.transformer.h[i].ln_2, model1.transformer.h[i].ln_2)
        self.model.transformer.ln_f = RMSMerger(model0.transformer.ln_f, model1.transformer.ln_f)
        self.model.lm_head = LinearMerger(model0.lm_head, model1.lm_head)

        self._sampler = lambda: 0.5

    def set_sampler(self, sampler_type, fixed_coeff=0.5):
        if sampler_type is None:
            self._sampler = lambda: fixed_coeff
        elif sampler_type == "narrow_uniform":
            import random as _random
            self._sampler = lambda: _random.uniform(0.4, 0.6)
        elif sampler_type == "uniform":
            import random as _random
            self._sampler = lambda: _random.uniform(0.0, 1.0)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type!r}")

    def _project(self, coeff):


        perm_type = "soft_perm" if (self._soft_perm and self.training) else "perm"
        P_res = project(self.proj["residual"], perm_type if self._permutations_only else "ortho")
        self.model.transformer.wte.set_P(P_res); self.model.transformer.wte.set_coeff(coeff)
        self.model.transformer.wpe.set_P(P_res); self.model.transformer.wpe.set_coeff(coeff)
        for i in range(len(self.model.transformer.h)):
            h = self.model.transformer.h[i]
            h.ln_1.set_P(P_res.t()); h.ln_1.set_coeff(coeff)
            h.attn.c_attn.set_P_in(P_res.t())
            P_heads = project(self.proj[f"attention_heads_{i}"], perm_type)
            h.attn.c_attn.set_P_out(P_heads); h.attn.c_attn.set_coeff(coeff)
            h.attn.c_proj.set_P_out(P_res); h.attn.c_proj.set_P_in(P_heads)
            h.attn.c_proj.set_coeff(coeff)
            h.mlp.c_fc.set_P_in(P_res.t())
            P_mlp = project(self.proj[f"mlp_{i}"], perm_type)
            h.mlp.c_fc.set_P_out(P_mlp); h.mlp.c_fc.set_coeff(coeff)
            h.mlp.c_proj.set_P_out(P_res); h.mlp.c_proj.set_P_in(P_mlp.t())
            h.mlp.c_proj.set_coeff(coeff)
            h.ln_2.set_P(P_res.t()); h.ln_2.set_coeff(coeff)
        self.model.transformer.ln_f.set_P(P_res.t()); self.model.transformer.ln_f.set_coeff(coeff)
        self.model.lm_head.set_P_in(P_res.t()); self.model.lm_head.set_coeff(coeff)

    def forward(self, input_ids=None, labels=None, **kwargs):
        coeff = self._sampler()
        self._project(coeff=coeff)
        return self.model(input_ids=input_ids, labels=labels, **kwargs)


@torch.no_grad()
def activation_matching(model0, model1, loader, device, max_batches=20, iterations=3):
    """Paper baseline (Verma & Elbayad 2024, Git Re-Basin style):
    PERMUTATIONS ONLY — residual, MLP, and head permutations computed from
    activation correlations via optimal transport. Correlation stats are
    accumulated incrementally (no giant activation dumps), so GPU/CPU
    memory stays small."""
    m0 = model0.to(device).eval()
    m1 = copy.deepcopy(model1).to(device).eval()
    n_layers = len(m0.transformer.h)
    num_heads = m0.transformer.h[0].attn.num_heads
    inner = m0.transformer.h[0].mlp.c_fc.nf
    d = m0.transformer.wte.weight.shape[1]

    class CorrAccum:
        """Accumulates E[x0 x1^T] style stats for corr(unit_i^0, unit_j^1)."""
        def __init__(self, dim, dev):
            self.n = 0
            self.s0 = torch.zeros(dim, device=dev); self.s1 = torch.zeros(dim, device=dev)
            self.q0 = torch.zeros(dim, device=dev); self.q1 = torch.zeros(dim, device=dev)
            self.s01 = torch.zeros(dim, dim, device=dev)

        def add(self, x0, x1):
            self.n += x0.shape[0]
            self.s0 += x0.sum(0); self.s1 += x1.sum(0)
            self.q0 += (x0 ** 2).sum(0); self.q1 += (x1 ** 2).sum(0)
            self.s01 += x0.t() @ x1

        def corr(self):
            n = max(self.n, 1)
            m0_ = self.s0 / n; m1_ = self.s1 / n
            v0 = (self.q0 / n - m0_ ** 2).clamp_min(1e-12).sqrt()
            v1 = (self.q1 / n - m1_ ** 2).clamp_min(1e-12).sqrt()
            cov = self.s01 / n - torch.outer(m0_, m1_)
            return cov / torch.outer(v0, v1)

    def run_with_hooks(model, x, store):
        """Forward pass collecting: ln_f input (residual), per-layer MLP
        pre-activations, per-layer per-head activation magnitudes."""
        handles = []

        def resid_hook(module, args):
            store["resid"] = args[0].mean(dim=1).float()

        handles.append(model.transformer.ln_f.register_forward_pre_hook(resid_hook))
        for li in range(n_layers):
            def fc_hook(module, args, output, li=li):
                store[f"mlp_{li}"] = output.reshape(-1, inner).float()

            def proj_hook(module, args, li=li):
                a = args[0]
                B, T, HD = a.shape
                a = a.view(B, T, num_heads, HD // num_heads)
                store[f"head_{li}"] = a.abs().mean(dim=(1, 3)).float()

            handles.append(model.transformer.h[li].mlp.c_fc.register_forward_hook(fc_hook))
            handles.append(model.transformer.h[li].attn.c_proj.register_forward_pre_hook(proj_hook))
        model(input_ids=x)
        for hd in handles:
            hd.remove()

    for it in range(iterations):

        resid_acc = CorrAccum(d, device)
        mlp_acc = [CorrAccum(inner, device) for _ in range(n_layers)]
        head_acc = [CorrAccum(num_heads, device) for _ in range(n_layers)]
        for bi, x in enumerate(loader):
            if bi >= max_batches:
                break
            s0, s1 = {}, {}
            run_with_hooks(m0, x, s0)
            run_with_hooks(m1, x, s1)
            resid_acc.add(s1["resid"], s0["resid"])
            for li in range(n_layers):
                mlp_acc[li].add(s0[f"mlp_{li}"], s1[f"mlp_{li}"])
                head_acc[li].add(s0[f"head_{li}"], s1[f"head_{li}"])


        m1 = m1.cpu()


        P_res = otify((-resid_acc.corr()).cpu())
        ortho_residual(m1, P_res)
        for li in range(n_layers):
            P_mlp = otify((-mlp_acc[li].corr()).cpu())
            permute_mlp(m1, li, P_mlp.t())
            P_heads = otify((-head_acc[li].corr()).cpu())
            permute_heads(m1, li, P_heads)
        m1 = m1.to(device).eval()
        print(f"[activation_matching] iter {it + 1}/{iterations} done")
    return m1


@torch.no_grad()
def evaluate_lm(model, loader, max_batches=None):
    """Return (loss, perplexity, token_accuracy%) on a token loader."""
    model.eval()
    loss_sum, tok_sum, correct, total = 0.0, 0, 0, 0
    for i, x in enumerate(loader):
        out = model(input_ids=x, labels=x)
        n_tok = x.shape[0] * (x.shape[1] - 1)
        loss_sum += out.loss.item() * n_tok
        tok_sum += n_tok
        preds = out.logits[:, :-1, :].argmax(-1)
        correct += (preds == x[:, 1:]).sum().item()
        total += n_tok
        if max_batches and i + 1 >= max_batches:
            break
    loss = loss_sum / tok_sum
    return loss, math.exp(min(loss, 20.0)), 100.0 * correct / total

@torch.no_grad()
def eval_merger_at(merger, coeff, loader, max_batches=None):
    """Project once at a fixed coeff, then evaluate merger.model directly
    (much faster than projecting inside every forward)."""
    merger.eval()
    merger._project(coeff=float(coeff))
    return evaluate_lm(merger.model, loader, max_batches=max_batches)

def interpolate_state_dicts(sd0, sd1, coeff):
    out = {}
    for k, v0 in sd0.items():
        v1 = sd1.get(k, None)
        if v1 is not None and torch.is_floating_point(v0) and torch.is_floating_point(v1):
            out[k] = coeff * v0 + (1 - coeff) * v1
        else:
            out[k] = v0
    return out

def summarize(coeff_losses):
    """max barrier, midpoint loss, per-coeff barriers (coeff=1 -> model0)."""
    L0 = coeff_losses[1.0]; L1 = coeff_losses[0.0]
    barriers, max_barrier = {}, -float("inf")
    for c, L in coeff_losses.items():
        barriers[c] = L - (c * L0 + (1 - c) * L1)
        max_barrier = max(max_barrier, barriers[c])
    return max_barrier, coeff_losses[0.5], barriers

print("[lib] official alignment code + activation matching + eval ready")


def _find_split_files(root, split):
    files = sorted(Path(root).rglob(f"python_{split}_*.jsonl.gz"))
    if not files:
        files = sorted(Path(root).rglob(f"python_{split}*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No python {split} jsonl files under {root}")
    return files

def _read_code_strings(files):
    texts = []
    for fp in files:
        opener = gzip.open if fp.suffix == ".gz" else open
        with opener(fp, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = rec.get("code") or rec.get("original_string") or rec.get("function")
                if code:
                    texts.append(code)
    return texts

def _tokenize_to_array(tokenizer, texts, batch=1000):
    chunks = []
    for s in range(0, len(texts), batch):
        enc = tokenizer(texts[s:s + batch])["input_ids"]
        chunks.append(np.concatenate([np.asarray(ids, dtype=np.uint16) for ids in enc])
                      if enc else np.zeros(0, dtype=np.uint16))
        if (s // batch) % 50 == 0:
            print(f"  tokenized {s + len(enc):,}/{len(texts):,} functions", end="\r")
    print()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)

class GPUTokenLoader:
    """Iterates (B, block) int64 batches sliced from a GPU-resident
    int32 chunk matrix. No CPU workers, no host memory pressure."""

    def __init__(self, chunks, batch_size, shuffle, drop_last=True):
        self.chunks = chunks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = self.chunks.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = self.chunks.shape[0]
        dev = self.chunks.device
        order = torch.randperm(n, device=dev) if self.shuffle else torch.arange(n, device=dev)
        end = n - (n % self.batch_size) if self.drop_last else n
        for s in range(0, end, self.batch_size):
            yield self.chunks[order[s:s + self.batch_size]].long()

def build_csn_data(root, device, block_size, batch_size, eval_batch,
                   max_train_funcs=None, cache_dir="csn_cache"):
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    arrays = {}
    for split in ("train", "valid", "test"):
        suffix = f"_{max_train_funcs}" if (split == "train" and max_train_funcs) else ""
        cpath = cache / f"python_{split}{suffix}_tokens.npy"
        if cpath.exists():
            arrays[split] = np.load(cpath)
        else:
            texts = _read_code_strings(_find_split_files(root, split))
            if split == "train" and max_train_funcs:
                texts = texts[:max_train_funcs]
            print(f"[data] {split}: {len(texts):,} functions -> tokenizing...")
            arr = _tokenize_to_array(tokenizer, texts)
            np.save(cpath, arr)
            arrays[split] = arr
        print(f"[data] {split}: {arrays[split].shape[0]:,} tokens (cache {cpath.name})")

    def to_chunks(arr):
        n = (arr.shape[0] // block_size) * block_size
        t = torch.from_numpy(arr[:n].astype(np.int32)).view(-1, block_size)
        return t.to(device)

    train_chunks = to_chunks(arrays["train"])
    valid_chunks = to_chunks(arrays["valid"])
    test_chunks = to_chunks(arrays["test"])
    print(f"[data] chunks(block={block_size}): train={train_chunks.shape[0]:,} "
          f"valid={valid_chunks.shape[0]:,} test={test_chunks.shape[0]:,}  (on {device})")


    token_freqs = torch.bincount(train_chunks.reshape(-1).long(),
                                 minlength=tokenizer.vocab_size).cpu()
    print(f"[data] token_freqs: {(token_freqs > 0).sum().item():,}/{tokenizer.vocab_size:,} tokens seen in train")

    trainloader = GPUTokenLoader(train_chunks, batch_size, shuffle=True, drop_last=True)
    validloader = GPUTokenLoader(valid_chunks, eval_batch, shuffle=False, drop_last=False)
    testloader = GPUTokenLoader(test_chunks, eval_batch, shuffle=False, drop_last=False)
    statloader = GPUTokenLoader(train_chunks, batch_size, shuffle=True, drop_last=True)
    return trainloader, validloader, testloader, statloader, tokenizer, token_freqs


cfg = {
    "dataset": "CodeSearchNet-Python",

    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "n_inner": 2048,
    "block_size": 512,


    "tie_word_embeddings": True,

    "epochs": 5,
    "batch_size": 64,
    "lr": 2.5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "grad_clip": 1.0,
}

SMOKE = False

MAX_TRAIN_FUNCS = None
EVAL_EVERY = 500
EVAL_MAX_BATCHES = 50
MERGER_STEPS = 1500
MERGER_LR = 3e-4
WM_ITERS = 15
ACT_BATCHES = 20
SWEEP_MAX_BATCHES = None
EVAL_BATCH = 64

if SMOKE:
    cfg["epochs"] = 1
    MAX_TRAIN_FUNCS = 2000
    EVAL_EVERY, EVAL_MAX_BATCHES = 5, 2
    MERGER_STEPS, WM_ITERS, ACT_BATCHES, SWEEP_MAX_BATCHES = 5, 2, 2, 2

print(f"[cfg] SMOKE={SMOKE}  epochs={cfg['epochs']}  merger_steps={MERGER_STEPS}  "
      f"wm_iters={WM_ITERS}")


OUT_DIR = Path("outputs_gpt2csn")
MODELS_DIR = OUT_DIR / "models"
LOGS_DIR = OUT_DIR / "logs"
MERGERS_DIR = OUT_DIR / "mergers"
PLOTS_DIR = OUT_DIR / "plots"
for _d in (MODELS_DIR, LOGS_DIR, MERGERS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
print(f"[out] saving everything under {OUT_DIR.resolve()}")


import kagglehub

raw_path = kagglehub.dataset_download("omduggineni/codesearchnet")
print("Path to dataset files:", raw_path)

trainloader, validloader, testloader, statloader, tokenizer, token_freqs = build_csn_data(
    raw_path,
    device,
    block_size=cfg["block_size"],
    batch_size=cfg["batch_size"],
    eval_batch=EVAL_BATCH,
    max_train_funcs=MAX_TRAIN_FUNCS,
)
(LOGS_DIR / "data_info.json").write_text(json.dumps({
    "dataset": "CodeSearchNet python (official splits)",
    "train_chunks": len(trainloader) * cfg["batch_size"],
    "block_size": cfg["block_size"],
    "max_train_funcs": MAX_TRAIN_FUNCS,
    "vocab_size": tokenizer.vocab_size,
}, indent=2))


(LOGS_DIR / "run_config.json").write_text(json.dumps(cfg, indent=2))
tokenizer.save_pretrained(str(LOGS_DIR.parent / "tokenizer"))


def build_gpt2(seed):
    torch.manual_seed(seed)
    configuration = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg["block_size"],
        n_ctx=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_inner=cfg["n_inner"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        activation_function="gelu_new",
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(configuration)


    return model

def train_one_gpt2(seed, trainloader, validloader, epochs,
                   quick_batches=None, eval_every=500, eval_max_batches=50):
    model = build_gpt2(seed).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  seed{seed}: GPT-2 with {n_params:.1f}M params "
          f"(tied={model.transformer.wte.weight is model.lm_head.weight})")


    run_model = model
    if device == "cuda":
        try:
            run_model = torch.compile(model)
        except Exception as _e:
            print(f"  torch.compile disabled: {_e}")
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = max(epochs * len(trainloader), 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = device == "cuda"

    history = []
    log_path = LOGS_DIR / f"gpt2_seed{seed}_train_log.json"
    ckpt_path = MODELS_DIR / f"gpt2_seed{seed}.pth"
    resume_path = MODELS_DIR / f"gpt2_seed{seed}_resume.pth"


    start_epoch = 0
    if log_path.exists() and ckpt_path.exists():
        history = json.loads(log_path.read_text())
        done = [h for h in history if h.get("kind") == "epoch"]
        start_epoch = min(len(done), epochs)
        if start_epoch > 0:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            if resume_path.exists():
                rs = torch.load(resume_path, map_location=device)
                optimizer.load_state_dict(rs["optimizer"])
                scheduler.load_state_dict(rs["scheduler"])
            print(f"  seed{seed}: resumed at epoch {start_epoch}/{epochs}")

    gstep = start_epoch * len(trainloader)
    for epoch in range(start_epoch, epochs):
        model.train()
        tr_loss, tr_tok, tr_correct = 0.0, 0, 0
        for i, x in enumerate(trainloader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp):
                out = run_model(input_ids=x, labels=x)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            gstep += 1
            n_tok = x.shape[0] * (x.shape[1] - 1)
            tr_loss += out.loss.item() * n_tok
            tr_tok += n_tok
            tr_correct += (out.logits[:, :-1, :].argmax(-1) == x[:, 1:]).sum().item()
            if gstep % eval_every == 0:
                vl, vppl, vacc = evaluate_lm(model, validloader, max_batches=eval_max_batches)
                model.train()
                history.append({"kind": "step", "step": gstep,
                                "train_loss": tr_loss / tr_tok,
                                "val_loss": vl, "val_ppl": vppl, "val_token_acc": vacc,
                                "lr": scheduler.get_last_lr()[0]})
                log_path.write_text(json.dumps(history, indent=2))
                print(f"  seed{seed} step {gstep}: train_loss={tr_loss / tr_tok:.4f} "
                      f"val_loss={vl:.4f} val_ppl={vppl:.2f} val_acc={vacc:.2f}%")
            if quick_batches and i + 1 >= quick_batches:
                break
        vl, vppl, vacc = evaluate_lm(model, validloader, max_batches=eval_max_batches)
        history.append({"kind": "epoch", "epoch": epoch + 1,
                        "train_loss": tr_loss / max(tr_tok, 1),
                        "train_token_acc": 100.0 * tr_correct / max(tr_tok, 1),
                        "val_loss": vl, "val_ppl": vppl, "val_token_acc": vacc,
                        "lr": scheduler.get_last_lr()[0]})
        log_path.write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), ckpt_path)
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1}, resume_path)
        print(f"  seed{seed} ep{epoch + 1}/{epochs}: train_loss={history[-1]['train_loss']:.4f} "
              f"train_acc={history[-1]['train_token_acc']:.2f}% "
              f"val_loss={vl:.4f} val_ppl={vppl:.2f} val_acc={vacc:.2f}%")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  saved {ckpt_path} and {log_path}")
    return model, history


print("=" * 20, "TRAIN GPT-2 seed=0", "=" * 20)
model0, history0 = train_one_gpt2(0, trainloader, validloader, cfg["epochs"],
                                  quick_batches=5 if SMOKE else None,
                                  eval_every=EVAL_EVERY, eval_max_batches=EVAL_MAX_BATCHES)


print("=" * 20, "TRAIN GPT-2 seed=1", "=" * 20)
model1, history1 = train_one_gpt2(1, trainloader, validloader, cfg["epochs"],
                                  quick_batches=5 if SMOKE else None,
                                  eval_every=EVAL_EVERY, eval_max_batches=EVAL_MAX_BATCHES)


print("### Endpoint metrics (held-out test split)")
_endpoint = {}
for _nm, _m in (("seed0", model0), ("seed1", model1)):
    _l, _p, _a = evaluate_lm(_m, testloader)
    _endpoint[_nm] = {"test_loss": _l, "test_ppl": _p, "test_token_acc": _a}
    print(f"  {_nm}: test_loss={_l:.4f}  ppl={_p:.2f}  token_acc={_a:.2f}%")
(LOGS_DIR / "endpoint_metrics.json").write_text(json.dumps(_endpoint, indent=2))


_coeffs = [round(0.1 * _i, 1) for _i in range(11)]
_template = copy.deepcopy(model0)
_sd0 = {k: v.detach().clone() for k, v in model0.state_dict().items()}
_sd1 = {k: v.detach().clone() for k, v in model1.state_dict().items()}
curve_van, ppl_van, acc_van = {}, {}, {}
with torch.no_grad():
    for _c in _coeffs:
        _template.load_state_dict(interpolate_state_dicts(_sd0, _sd1, _c))
        _l, _p, _a = evaluate_lm(_template, testloader, max_batches=SWEEP_MAX_BATCHES)
        curve_van[_c], ppl_van[_c], acc_van[_c] = _l, _p, _a
        print(f"[Vanilla] λ={_c} -> loss={_l:.4f} ppl={_p:.2f} acc={_a:.2f}%")
del _template, _sd0, _sd1
(LOGS_DIR / "curve_vanilla.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_van.items()},
     "ppl": {str(k): v for k, v in ppl_van.items()},
     "acc": {str(k): v for k, v in acc_van.items()}}, indent=2))


c0 = canonicalize(copy.deepcopy(model0).cpu().float().eval()).to(device).eval()
c1 = canonicalize(copy.deepcopy(model1).cpu().float().eval()).to(device).eval()
print("[canonical] c0, c1 ready (RMSNorm + circuits, function preserving)")


print("=" * 20, "ACTIVATION MATCHING", "=" * 20)
_a1 = activation_matching(c0, c1, statloader, device,
                          max_batches=ACT_BATCHES, iterations=3)
torch.save(_a1.state_dict(), MODELS_DIR / "gpt2_seed1_activation_matched.pth")

_coeffs = [round(0.1 * _i, 1) for _i in range(11)]
_template = copy.deepcopy(c0)
_sd0 = {k: v.detach().clone() for k, v in c0.state_dict().items()}
_sd1 = {k: v.detach().clone() for k, v in _a1.state_dict().items()}
curve_act, ppl_act, acc_act = {}, {}, {}
with torch.no_grad():
    for _c in _coeffs:
        _template.load_state_dict(interpolate_state_dicts(_sd0, _sd1, _c))
        _l, _p, _a = evaluate_lm(_template, testloader, max_batches=SWEEP_MAX_BATCHES)
        curve_act[_c], ppl_act[_c], acc_act[_c] = _l, _p, _a
        print(f"[Activation] λ={_c} -> loss={_l:.4f} ppl={_p:.2f} acc={_a:.2f}%")
del _template, _sd0, _sd1, _a1
(LOGS_DIR / "curve_activation_matching.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_act.items()},
     "ppl": {str(k): v for k, v in ppl_act.items()},
     "acc": {str(k): v for k, v in acc_act.items()}}, indent=2))


print("=" * 20, "WEIGHT MATCHING (OURS)", "=" * 20)
torch.manual_seed(42)
random.seed(42)
_wm = GPTMerger(copy.deepcopy(model0).cpu().float(),
                copy.deepcopy(model1).cpu().float(),
                token_freqs=token_freqs, permutations_only=False,
                iterations=WM_ITERS).to(device)
_coeffs = [round(0.1 * _i, 1) for _i in range(11)]
curve_wm, ppl_wm, acc_wm = {}, {}, {}
for _c in _coeffs:
    _l, _p, _a = eval_merger_at(_wm, _c, testloader, max_batches=SWEEP_MAX_BATCHES)
    curve_wm[_c], ppl_wm[_c], acc_wm[_c] = _l, _p, _a
    print(f"[Weight matching] λ={_c} -> loss={_l:.4f} ppl={_p:.2f} acc={_a:.2f}%")
del _wm
(LOGS_DIR / "curve_weight_matching.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_wm.items()},
     "ppl": {str(k): v for k, v in ppl_wm.items()},
     "acc": {str(k): v for k, v in acc_wm.items()}}, indent=2))


def train_gpt2_merger(model0, model1, trainloader, validloader, *, name,
                      permutations_only, steps, wm_iters, lr,
                      eval_every=250, eval_max_batches=50, seed=42,
                      soft_perm=False):
    torch.manual_seed(seed)
    random.seed(seed)
    merger = GPTMerger(copy.deepcopy(model0).cpu().float(),
                       copy.deepcopy(model1).cpu().float(),
                       token_freqs=token_freqs,
                       permutations_only=permutations_only,
                       iterations=wm_iters, soft_perm=soft_perm).to(device)
    merger.set_sampler("narrow_uniform")
    trainable = [p for p in merger.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * steps), steps)

    history = []
    log_path = LOGS_DIR / f"merger_{name}_train_log.json"
    ckpt_path = MERGERS_DIR / f"merger_{name}.pth"
    resume_path = MERGERS_DIR / f"merger_{name}_resume.pth"


    start_step = 0
    if log_path.exists() and ckpt_path.exists():
        history = json.loads(log_path.read_text())
        trained = [h["step"] for h in history if h.get("step", 0) > 0]
        start_step = min(max(trained) if trained else 0, steps)
        if start_step > 0:
            merger.load_state_dict(torch.load(ckpt_path, map_location=device))
            if resume_path.exists():
                rs = torch.load(resume_path, map_location=device)
                optimizer.load_state_dict(rs["optimizer"])
                scheduler.load_state_dict(rs["scheduler"])
            print(f"  merger {name}: resumed at step {start_step}/{steps}")

    if not history:
        vl, vppl, vacc = eval_merger_at(merger, 0.5, validloader, max_batches=eval_max_batches)
        history.append({"step": 0, "mid_val_loss": vl, "mid_val_ppl": vppl, "mid_val_acc": vacc})
        log_path.write_text(json.dumps(history, indent=2))
        print(f"  weight-matched midpoint: val_loss={vl:.4f} ppl={vppl:.2f} acc={vacc:.2f}%")

    def batches():
        while True:
            for xb in trainloader:
                yield xb

    gen = batches()
    merger.train()
    run_loss, run_tok = 0.0, 0
    for step in range(start_step + 1, steps + 1):
        x = next(gen)
        optimizer.zero_grad(set_to_none=True)
        out = merger(input_ids=x, labels=x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        n_tok = x.shape[0] * (x.shape[1] - 1)
        run_loss += out.loss.item() * n_tok
        run_tok += n_tok
        if step % eval_every == 0 or step == steps:
            vl, vppl, vacc = eval_merger_at(merger, 0.5, validloader, max_batches=eval_max_batches)
            merger.train()
            history.append({"step": step, "train_loss": run_loss / max(run_tok, 1),
                            "mid_val_loss": vl, "mid_val_ppl": vppl, "mid_val_acc": vacc,
                            "lr": scheduler.get_last_lr()[0]})
            log_path.write_text(json.dumps(history, indent=2))
            torch.save(merger.state_dict(), ckpt_path)
            torch.save({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step}, resume_path)
            run_loss, run_tok = 0.0, 0
            with torch.no_grad():
                _P = merger.proj["residual"]
                _moved = int((_P.argmax(1) != torch.arange(_P.shape[0], device=_P.device)).sum())
            print(f"  {name} step {step}/{steps}: mid_val_loss={vl:.4f} "
                  f"ppl={vppl:.2f} acc={vacc:.2f}%  "
                  f"residual assignments moved: {_moved}/{_P.shape[0]}")
    torch.save(merger.state_dict(), ckpt_path)
    print(f"  saved {ckpt_path} and {log_path}")
    return merger


print("=" * 20, "LEARNED MATCHING (OURS)", "=" * 20)
_merger_lm = train_gpt2_merger(model0, model1, trainloader, validloader,
                               name="ours", permutations_only=False,
                               steps=MERGER_STEPS, wm_iters=WM_ITERS, lr=MERGER_LR,
                               eval_max_batches=EVAL_MAX_BATCHES)
_coeffs = [round(0.1 * _i, 1) for _i in range(11)]
curve_lm, ppl_lm, acc_lm = {}, {}, {}
for _c in _coeffs:
    _l, _p, _a = eval_merger_at(_merger_lm, _c, testloader, max_batches=SWEEP_MAX_BATCHES)
    curve_lm[_c], ppl_lm[_c], acc_lm[_c] = _l, _p, _a
    print(f"[Learned (ours)] λ={_c} -> loss={_l:.4f} ppl={_p:.2f} acc={_a:.2f}%")
del _merger_lm
(LOGS_DIR / "curve_learned_matching_ours.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_lm.items()},
     "ppl": {str(k): v for k, v in ppl_lm.items()},
     "acc": {str(k): v for k, v in acc_lm.items()}}, indent=2))


print("=" * 20, "LEARNED MATCHING (PERMUTATIONS)", "=" * 20)


FORCE_RETRAIN_LMP = True
if FORCE_RETRAIN_LMP:
    for _stale in (
        MERGERS_DIR / "merger_permutations.pth",
        MERGERS_DIR / "merger_permutations_resume.pth",
        LOGS_DIR / "merger_permutations_train_log.json",
        LOGS_DIR / "curve_learned_matching_permutations.json",
    ):
        if _stale.exists():
            _stale.unlink()
            print(f"[clean] removed stale {_stale.name}")


_merger_lmp = train_gpt2_merger(model0, model1, trainloader, validloader,
                                name="permutations", permutations_only=True,
                                steps=MERGER_STEPS, wm_iters=WM_ITERS, lr=MERGER_LR,
                                eval_max_batches=EVAL_MAX_BATCHES, soft_perm=True)
_coeffs = [round(0.1 * _i, 1) for _i in range(11)]
curve_lmp, ppl_lmp, acc_lmp = {}, {}, {}
for _c in _coeffs:
    _l, _p, _a = eval_merger_at(_merger_lmp, _c, testloader, max_batches=SWEEP_MAX_BATCHES)
    curve_lmp[_c], ppl_lmp[_c], acc_lmp[_c] = _l, _p, _a
    print(f"[Learned (perm)] λ={_c} -> loss={_l:.4f} ppl={_p:.2f} acc={_a:.2f}%")
del _merger_lmp
(LOGS_DIR / "curve_learned_matching_permutations.json").write_text(json.dumps(
    {"loss": {str(k): v for k, v in curve_lmp.items()},
     "ppl": {str(k): v for k, v in ppl_lmp.items()},
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
ppl_curves = {
    "Vanilla averaging": ppl_van,
    "Activation matching": ppl_act,
    "Weight matching (ours)": ppl_wm,
    "Learned matching (permutations)": ppl_lmp,
    "Learned matching (ours)": ppl_lm,
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
    ax.set_title("CodeSearchNet-Python · LMC of two GPT-2s")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.subplots_adjust(right=0.62)
    fig.savefig(PLOTS_DIR / fname, dpi=200, bbox_inches="tight", facecolor="white")
    return fig

fig_loss = _lambda_plot(curves, "Test loss", "gpt2csn_loss_vs_lambda.png")
fig_acc = _lambda_plot(acc_curves, "Test next-token accuracy (%)", "gpt2csn_acc_vs_lambda.png")


fig_tr, ax_tr = plt.subplots(figsize=(7, 4))
for _sd, _hist, _col in (("seed0", history0, "#1f77b4"), ("seed1", history1, "#d62728")):
    _ep = [h for h in _hist if h.get("kind") == "epoch"]
    ax_tr.plot([h["epoch"] for h in _ep], [h["train_loss"] for h in _ep],
               color=_col, linestyle="--", label=f"{_sd} train")
    ax_tr.plot([h["epoch"] for h in _ep], [h["val_loss"] for h in _ep],
               color=_col, label=f"{_sd} val")
ax_tr.set_xlabel("Epoch"); ax_tr.set_ylabel("Loss"); ax_tr.legend(fontsize=8); ax_tr.grid(True)
ax_tr.set_title("CodeSearchNet-Python · GPT-2 training (seeds 0 & 1)")
fig_tr.tight_layout()
fig_tr.savefig(PLOTS_DIR / "gpt2csn_training_curves.png", dpi=200,
               bbox_inches="tight", facecolor="white")


rows = []
for name in curves:
    mb, mid, _ = summarize(curves[name])
    rows.append({"method": name, "max_barrier": mb, "mid_loss": mid,
                 "mid_ppl": ppl_curves[name][0.5], "mid_token_acc": acc_curves[name][0.5],
                 "endpoints_loss": (curves[name][0.0] + curves[name][1.0]) / 2})
rows.sort(key=lambda r: r["max_barrier"], reverse=True)

_hdr = f"{'Method':34s} | {'Barrier':>8s} | {'Mid loss':>8s} | {'Mid PPL':>8s} | {'Mid acc%':>8s}"
print("=" * len(_hdr)); print(_hdr); print("-" * len(_hdr))
for r in rows:
    print(f"{r['method']:34s} | {r['max_barrier']:8.3f} | {r['mid_loss']:8.3f} | "
          f"{r['mid_ppl']:8.2f} | {r['mid_token_acc']:8.2f}")
print("=" * len(_hdr))

table_md = "| Method | Loss barrier | Mid loss | Mid PPL | Mid token acc |\n"
table_md += "|---|---|---|---|---|\n"
for r in rows:
    _b = "**" if r["method"] == "Learned matching (ours)" else ""
    table_md += (f"| {_b}{r['method']}{_b} | {_b}{r['max_barrier']:.3f}{_b} | "
                 f"{r['mid_loss']:.3f} | {r['mid_ppl']:.2f} | {r['mid_token_acc']:.2f}% |\n")

serial = {
    "loss_curves": {k: {str(c): float(v) for c, v in d.items()} for k, d in curves.items()},
    "ppl_curves": {k: {str(c): float(v) for c, v in d.items()} for k, d in ppl_curves.items()},
    "acc_curves": {k: {str(c): float(v) for c, v in d.items()} for k, d in acc_curves.items()},
    "table": rows,
}
(OUT_DIR / "gpt2csn_interp_curves.json").write_text(json.dumps(serial, indent=2))
print(f"[save] plots -> {PLOTS_DIR}  curves+table -> {OUT_DIR / 'gpt2csn_interp_curves.json'}")


import zipfile

def _zip_results():
    zip_path = Path("gpt2csn_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(Path("outputs_gpt2csn").rglob("*")):
            if f.is_file():
                zf.write(f, f.as_posix())
    print(f"[zip] {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path.read_bytes()

results_download
