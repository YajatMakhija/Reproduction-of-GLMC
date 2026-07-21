"""Data-free weight matching: align model1 to model0 in place.

Estimates and applies to model1: a global **orthogonal** rotation of the residual
stream (closed-form Procrustes / SVD) and per-layer **permutations** of the
attention heads and MLP neurons (exact optimal transport). All transforms are
function preserving, so model1's own loss is unchanged. Iterated to convergence.
"""

import ot
import torch
from einops import rearrange


# --------------------------------------------------------------------------- #
# Residual-stream orthogonal alignment
# --------------------------------------------------------------------------- #
def compute_optimal_orthogonal_matrix(t1, t2):
    """Orthogonal O minimizing ||t1 - t2 @ O||_F  (O = U Vt from SVD of t2^T t1)."""
    U, _, Vh = torch.linalg.svd(t2.T @ t1)
    return U @ Vh


def _rotate(x, O):
    return x @ O.t() if x is not None else None


def ortho_residual(model, O):
    """Apply residual rotation O to every residual-touching weight/bias/offset."""
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
        layer[0].offset.data = _rotate(layer[0].offset.data, O)   # pre-attn norm offset
        layer[2].offset.data = _rotate(layer[2].offset.data, O)   # pre-ff norm offset

    model.final_norm.offset.data = _rotate(model.final_norm.offset.data, O)
    model.linear_head.weight.data = model.linear_head.weight.data @ O.t()
    return model


def _residual_stack(model0, model1, include_blocks):
    """Stack residual-facing weights (each model's own), column-normalized."""
    a = [model0.to_patch_embedding[2].weight.data, model0.pos_embedding.weight.data.t(), model0.linear_head.weight.data.t()]
    b = [model1.to_patch_embedding[2].weight.data, model1.pos_embedding.weight.data.t(), model1.linear_head.weight.data.t()]
    if include_blocks:
        for l0, l1 in zip(model0.transformer.layers, model1.transformer.layers):
            a += [l0[1].to_qkv.weight.data.t(), l0[1].to_out[0].weight.data, l0[3].net[0].weight.data.t(), l0[3].net[3].weight.data]
            b += [l1[1].to_qkv.weight.data.t(), l1[1].to_out[0].weight.data, l1[3].net[0].weight.data.t(), l1[3].net[3].weight.data]
    a = [t / t.shape[1] ** 0.5 for t in a]
    b = [t / t.shape[1] ** 0.5 for t in b]
    return torch.cat(a, dim=1).t(), torch.cat(b, dim=1).t()


# --------------------------------------------------------------------------- #
# Head / MLP permutations via optimal transport
# --------------------------------------------------------------------------- #
def otify(cost):
    """Exact OT with uniform marginals == a permutation matrix."""
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
    """QK and OV circuits; the output projection is taken from out_model (=model1)."""
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
    if net[0].bias is not None:           # mlp hidden reader bias -> permute
        net[0].bias.data = P @ net[0].bias.data
    return model


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def weight_matching(model0, model1, heads, iterations=15):
    """Align model1 to model0 in place; returns model1."""
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
