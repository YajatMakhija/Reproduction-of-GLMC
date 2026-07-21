"""Numerical helpers: interpolation, manifold projection, attention circuits."""

import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from enums import MatrixType


def interpolate(W0, W1, coeff):
    """coeff * W0 + (1 - coeff) * W1  (coeff=1 -> model0, coeff=0 -> model1)."""
    return coeff * W0 + (1 - coeff) * W1


def _make_orthogonal(A):
    U, _, Vt = torch.linalg.svd(A.cpu(), full_matrices=False)
    return (U @ Vt).to(A.device)


def _make_permutation(A):
    row, col = linear_sum_assignment(-A.detach().cpu().numpy())
    P = torch.zeros_like(A)
    P[row, col] = 1
    return P


def project(A, matrix_type):
    """Project A onto a matrix manifold. PERM uses a straight-through estimator."""
    if matrix_type == MatrixType.PERM:
        P = _make_permutation(A)
        return P.detach() + (A - A.detach())
    if matrix_type == MatrixType.ORTHO:
        return _make_orthogonal(A)
    raise ValueError(f"Unknown matrix type: {matrix_type}")


def project_to_attn_circuits(model, heads, dim, layer_i):
    """Rewrite one attention block into canonical QK / OV circuit form (K=V=I per
    head, Q/OUT hold the full circuits). Function preserving; makes head matching
    well posed and removes the intra-head symmetry."""
    Q, K, V = model.transformer.layers[layer_i][1].to_qkv.weight.data.chunk(3, dim=0)
    Q = rearrange(Q, "(h d) m -> h d m", h=heads, m=dim)
    K = rearrange(K, "(h d) m -> h d m", h=heads, m=dim)
    V = rearrange(V, "(h d) m -> h d m", h=heads, m=dim)
    OUT = rearrange(model.transformer.layers[layer_i][1].to_out[0].weight.data,
                    "m (h d) -> m h d", h=heads, m=dim).permute(1, 2, 0)
    QK = torch.bmm(Q.transpose(1, 2), K)
    OUTV = OUT.transpose(1, 2) @ V

    Q_new = torch.zeros_like(QK)
    K_new = torch.zeros_like(QK)
    V_new = torch.zeros_like(QK)
    OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1], device=QK.device)
    for h in range(QK.size(0)):
        eye = torch.eye(QK[h].shape[0], device=QK.device)
        Q_new[h] = QK[h].t();  K_new[h] = eye
        OUT_new[h] = OUTV[h].t();  V_new[h] = eye

    model.transformer.layers[layer_i][1].to_qkv.weight.data = torch.cat(
        (Q_new.reshape(-1, dim), K_new.reshape(-1, dim), V_new.reshape(-1, dim)), dim=0)
    model.transformer.layers[layer_i][1].to_out[0].weight.data = OUT_new.permute(2, 0, 1).reshape(dim, -1)
    return model
