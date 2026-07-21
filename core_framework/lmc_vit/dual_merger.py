"""Dual learned matching: a differentiable model that interpolates two aligned models.

This is the dual-sided counterpart of ``merger.py``, implementing "Dual Learned
Matching" (LMC-DM) from Li & Shen, *Scaling Linear Mode Connectivity and Merging
to Billion Parameter Pretrained Transformers*. The single-sided version freezes
model0's weights as the fixed target coordinate system and only learns latents
that rotate/permute model1 onto it:

    Theta_interp = coeff * Theta_A + (1 - coeff) * pi(Theta_B)

which forces model B to fully absorb the alignment cost. Dual learned matching
instead gives *both* endpoints their own learnable transformation and lets them
meet in a shared coordinate system:

    Theta_interp = coeff * pi_A(Theta_A) + (1 - coeff) * pi_B(Theta_B)

``ViTMerger`` wraps two weight-matched ``ReparamViT``s. Both models' raw weights
are frozen buffers; only the latent alignment matrices in ``self.proj_A`` and
``self.proj_B`` train (one orthogonal residual matrix + per-layer head/MLP
permutations, *per side*). Each forward projects those latents onto their
manifolds and wires them into per-module mergers that compute
``interpolate(pi_A(W0), pi_B(W1), coeff)``. Training both sets of latents to
minimize the interpolated model's loss drives the loss barrier to zero while
letting each model move toward a shared low-loss basin instead of anchoring one
side to the other's fixed coordinates (see Fig. 2 / Sec. 3.3 of the paper).
"""

import copy

import torch
from torch import nn
from einops import rearrange

from enums import MatrixType
from models import RMSNorm
from utils import interpolate, project, project_to_attn_circuits


# --------------------------------------------------------------------------- #
# Per-module mergers (dual-sided: both w0 and w1 carry learnable transforms)
# --------------------------------------------------------------------------- #
class LinearMerger(nn.Module):
    """interpolate(P_in_A @ W0 @ P_out_A, P_in_B @ W1 @ P_out_B); biases likewise."""

    def __init__(self, linear0, linear1):
        super().__init__()
        self.register_buffer("w0", linear0.weight.data.t().clone().contiguous())
        self.register_buffer("w1", linear1.weight.data.t().clone().contiguous())
        self.has_bias = linear0.bias is not None and linear1.bias is not None
        if self.has_bias:
            self.register_buffer("b0", linear0.bias.data.clone())
            self.register_buffer("b1", linear1.bias.data.clone())
        self.P_in_A = self.P_out_A = None
        self.P_in_B = self.P_out_B = None
        self.nf = self.w0.shape[1]

    def set_P_in_A(self, P):  self.P_in_A = P
    def set_P_out_A(self, P): self.P_out_A = P
    def set_P_in_B(self, P):  self.P_in_B = P
    def set_P_out_B(self, P): self.P_out_B = P

    def forward(self, x, coeff=0.5):
        size_out = x.size()[:-1] + (self.nf,)
        w_A = self.P_in_A @ self.w0 @ self.P_out_A
        w_B = self.P_in_B @ self.w1 @ self.P_out_B
        weight = interpolate(w_A, w_B, coeff)
        x = torch.matmul(x.view(-1, x.size(-1)), weight)
        if self.has_bias:
            x = x + interpolate(self.b0 @ self.P_out_A, self.b1 @ self.P_out_B, coeff)
        return x.view(size_out)


class EmbeddingMerger(nn.Module):
    def __init__(self, embedding0, embedding1):
        super().__init__()
        self.embedding_0 = copy.deepcopy(embedding0)
        self.embedding_1 = copy.deepcopy(embedding1)
        for p in self.parameters():
            p.requires_grad = False
        self.P_A = self.P_B = None

    def set_P_A(self, P): self.P_A = P
    def set_P_B(self, P): self.P_B = P

    def forward(self, x, coeff=0.5):
        return interpolate(self.embedding_0(x) @ self.P_A, self.embedding_1(x) @ self.P_B, coeff)


class RMSMerger(nn.Module):
    """RMSNorm merger; interpolates the additive offsets (each transformed by its own P)."""

    def __init__(self, rmsnorm0, rmsnorm1, dim):
        super().__init__()
        off0, off1 = rmsnorm0.offset, rmsnorm1.offset
        self.has_bias = off0 is not None and off1 is not None
        if self.has_bias:
            self.register_buffer("bias_0", off0.data.clone())
            self.register_buffer("bias_1", off1.data.clone())
        self.norm = RMSNorm(dim, eps=rmsnorm0.eps)
        self.P_A = self.P_B = None

    def set_P_A(self, P): self.P_A = P
    def set_P_B(self, P): self.P_B = P

    def forward(self, x, coeff=0.5):
        x = self.norm(x)
        if self.has_bias:
            return x + interpolate(self.bias_0 @ self.P_A.t(), self.bias_1 @ self.P_B.t(), coeff)
        return x


class LayerNormMerger(nn.Module):
    """Patch-embed LayerNorm merger (off the residual stream, no permutation needed
    on either side, since it operates on the raw patch-pixel dimension shared by
    both models rather than the aligned residual coordinate system)."""

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
        x = self.patch_embedding[0](x)          # Rearrange
        x = self.patch_embedding[1](x, coeff)   # LayerNormMerger
        x = self.patch_embedding[2](x, coeff)   # LinearMerger
        return self.patch_embedding[3](x)       # Dropout


class FeedForwardMerger(nn.Module):
    def __init__(self, feedforward):
        super().__init__()
        self.feedforward = feedforward

    def __getitem__(self, idx): return self.feedforward[idx]

    def forward(self, x, coeff):
        x = self.feedforward[0](x, coeff)   # LinearMerger (fc)
        x = self.feedforward[1](x)          # GELU
        x = self.feedforward[2](x)          # Dropout
        x = self.feedforward[3](x, coeff)   # LinearMerger (proj)
        return self.feedforward[4](x)       # Dropout


class AttnQKVMerger(nn.Module):
    """to_qkv merger. P_in_{A,B} = residual^T (input dim) for each side;
    P_out_{A,B} = the head permutation learned for each side."""

    def __init__(self, qkv0, qkv1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("w0", qkv0.weight.data.t().clone().contiguous())
        self.register_buffer("w1", qkv1.weight.data.t().clone().contiguous())
        self.P_in_A = self.P_out_A = None
        self.P_in_B = self.P_out_B = None
        self.nf = self.w0.shape[1]
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def set_P_in_A(self, P):  self.P_in_A = P
    def set_P_out_A(self, P): self.P_out_A = P
    def set_P_in_B(self, P):  self.P_in_B = P
    def set_P_out_B(self, P): self.P_out_B = P

    def _permute_heads(self, weight, P_in, P_head):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        Q, K, V = weight.t().chunk(3, dim=0)
        Q = Q @ P_in.t()
        Q = rearrange(Q, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim)
        K = rearrange(K, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim)
        V = rearrange(V, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim)
        Q = torch.bmm(Q.transpose(1, 2), P_in.t().expand(self.num_heads, -1, -1)).transpose(1, 2)
        Q, K, V = permute(Q, P_head), permute(K, P_head), permute(V, P_head)
        return torch.cat((Q.reshape(-1, self.embed_dim), K.reshape(-1, self.embed_dim),
                          V.reshape(-1, self.embed_dim)), dim=0).t()

    def forward(self, x, coeff=0.5):
        size_out = x.size()[:-1] + (self.nf,)
        w_A = self._permute_heads(self.w0, self.P_in_A, self.P_out_A)
        w_B = self._permute_heads(self.w1, self.P_in_B, self.P_out_B)
        weight = interpolate(w_A, w_B, coeff)
        return torch.mm(x.view(-1, x.size(-1)), weight).view(size_out)


class AttnOutMerger(nn.Module):
    """to_out[0] merger. P_in_{A,B} = the head permutation for each side;
    P_out_{A,B} = the residual rotation learned for each side."""

    def __init__(self, out0, out1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("w0", out0.weight.data.t().clone().contiguous())
        self.register_buffer("w1", out1.weight.data.t().clone().contiguous())
        self.has_bias = out0.bias is not None and out1.bias is not None
        if self.has_bias:
            self.register_buffer("b0", out0.bias.data.clone())
            self.register_buffer("b1", out1.bias.data.clone())
        self.P_in_A = self.P_out_A = None
        self.P_in_B = self.P_out_B = None
        self.nf = self.w0.shape[1]
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def set_P_in_A(self, P):  self.P_in_A = P
    def set_P_out_A(self, P): self.P_out_A = P
    def set_P_in_B(self, P):  self.P_in_B = P
    def set_P_out_B(self, P): self.P_out_B = P

    def _permute_heads(self, weight, P_out, P_head):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        OUT = rearrange(weight.t(), "m (h d) -> m h d", h=self.num_heads, m=self.embed_dim).permute(1, 2, 0)
        OUT = (OUT.transpose(1, 2) @ P_out.expand(self.num_heads, -1, -1)).transpose(1, 2)
        OUT = permute(OUT, P_head)
        return OUT.permute(2, 0, 1).reshape(self.embed_dim, -1).t()

    def forward(self, x, coeff=0.5):
        size_out = x.size()[:-1] + (self.nf,)
        w_A = self._permute_heads(self.w0 @ self.P_out_A, self.P_out_A, self.P_in_A)
        w_B = self._permute_heads(self.w1 @ self.P_out_B, self.P_out_B, self.P_in_B)
        weight = interpolate(w_A, w_B, coeff)
        x = torch.mm(x.view(-1, x.size(-1)), weight)
        if self.has_bias:
            x = x + interpolate(self.b0 @ self.P_out_A, self.b1 @ self.P_out_B, coeff)
        return x.view(size_out)


# --------------------------------------------------------------------------- #
# Full merger
# --------------------------------------------------------------------------- #
class ViTMerger(nn.Module):
    """Dual learned matching over two ``ReparamViT``s.

    Same public API as the single-sided ``merger.ViTMerger``
    (``__init__(model0, model1, num_heads, device)`` and
    ``forward(img, coeff)``), so it drops straight into an existing training
    loop. Internally, every latent alignment matrix in the single-sided version
    is duplicated into a matching pair (``self.proj_A`` / ``self.proj_B``), one
    per endpoint, both identity-initialized and both optimized jointly against
    the interpolation loss.
    """

    def __init__(self, model0, model1, num_heads, device="cpu"):
        super().__init__()
        model0, model1 = model0.eval(), model1.eval()
        embed_dim = model0.pos_embedding.weight.shape[1]
        assert embed_dim == model1.pos_embedding.weight.shape[1], "equal-width models only"
        patch_dim = model0.to_patch_embedding[2].weight.shape[1]
        n_layers = len(model0.transformer.layers)

        for i in range(n_layers):
            project_to_attn_circuits(model0, num_heads, embed_dim, i)
            project_to_attn_circuits(model1, num_heads, embed_dim, i)

        # latent alignment matrices (the only trainable parameters), identity
        # init on *both* sides -- this is the core change vs. single-sided
        # learned matching, which only allocates one such ParameterDict.
        self.proj_A = nn.ParameterDict({"residual": nn.Parameter(torch.eye(embed_dim))})
        self.proj_B = nn.ParameterDict({"residual": nn.Parameter(torch.eye(embed_dim))})
        for i in range(n_layers):
            self.proj_A[f"attention_heads_{i}"] = nn.Parameter(torch.eye(num_heads))
            self.proj_B[f"attention_heads_{i}"] = nn.Parameter(torch.eye(num_heads))
            mlp_dim = model0.transformer.layers[i][3].net[0].weight.shape[0]
            self.proj_A[f"mlp_{i}"] = nn.Parameter(torch.eye(mlp_dim))
            self.proj_B[f"mlp_{i}"] = nn.Parameter(torch.eye(mlp_dim))

        self.model = copy.deepcopy(model0)
        self.model.pos_embedding = EmbeddingMerger(model0.pos_embedding, model1.pos_embedding)
        self.model.to_patch_embedding[2] = LinearMerger(model0.to_patch_embedding[2], model1.to_patch_embedding[2])
        # patch pixels aren't part of the residual stream on either side, so the
        # read-in transform stays fixed identity for both A and B.
        self.model.to_patch_embedding[2].set_P_in_A(torch.eye(patch_dim, device=device))
        self.model.to_patch_embedding[2].set_P_in_B(torch.eye(patch_dim, device=device))
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
        # output/class space is shared and unambiguous -- fixed identity, both sides.
        num_classes = model0.linear_head.weight.shape[0]
        self.model.linear_head.set_P_out_A(torch.eye(num_classes, device=device))
        self.model.linear_head.set_P_out_B(torch.eye(num_classes, device=device))

    def _project(self):
        Q_A, _ = torch.linalg.qr(self.proj_A["residual"])
        Q_B, _ = torch.linalg.qr(self.proj_B["residual"])
        P_res_A, P_res_t_A = Q_A, Q_A.t()
        P_res_B, P_res_t_B = Q_B, Q_B.t()

        self.model.pos_embedding.set_P_A(P_res_A)
        self.model.pos_embedding.set_P_B(P_res_B)
        self.model.to_patch_embedding[2].set_P_out_A(P_res_A)
        self.model.to_patch_embedding[2].set_P_out_B(P_res_B)

        for i in range(len(self.model.transformer.layers)):
            layer = self.model.transformer.layers[i]

            P_mlp_A = project(self.proj_A[f"mlp_{i}"], MatrixType.PERM)
            P_mlp_B = project(self.proj_B[f"mlp_{i}"], MatrixType.PERM)
            layer[3].net[0].set_P_in_A(P_res_t_A);  layer[3].net[0].set_P_out_A(P_mlp_A)
            layer[3].net[0].set_P_in_B(P_res_t_B);  layer[3].net[0].set_P_out_B(P_mlp_B)
            layer[3].net[3].set_P_in_A(P_mlp_A.t()); layer[3].net[3].set_P_out_A(P_res_A)
            layer[3].net[3].set_P_in_B(P_mlp_B.t()); layer[3].net[3].set_P_out_B(P_res_B)

            P_heads_A = project(self.proj_A[f"attention_heads_{i}"], MatrixType.PERM)
            P_heads_B = project(self.proj_B[f"attention_heads_{i}"], MatrixType.PERM)
            layer[1].to_qkv.set_P_in_A(P_res_t_A);    layer[1].to_qkv.set_P_out_A(P_heads_A)
            layer[1].to_qkv.set_P_in_B(P_res_t_B);    layer[1].to_qkv.set_P_out_B(P_heads_B)
            layer[1].to_out[0].set_P_in_A(P_heads_A); layer[1].to_out[0].set_P_out_A(P_res_A)
            layer[1].to_out[0].set_P_in_B(P_heads_B); layer[1].to_out[0].set_P_out_B(P_res_B)

            layer[0].set_P_A(P_res_t_A); layer[0].set_P_B(P_res_t_B)
            layer[2].set_P_A(P_res_t_A); layer[2].set_P_B(P_res_t_B)

        self.model.final_norm.set_P_A(P_res_t_A); self.model.final_norm.set_P_B(P_res_t_B)
        self.model.linear_head.set_P_in_A(P_res_t_A); self.model.linear_head.set_P_in_B(P_res_t_B)

    def forward(self, img, coeff):
        self._project()
        return self.model(img, coeff=coeff)
