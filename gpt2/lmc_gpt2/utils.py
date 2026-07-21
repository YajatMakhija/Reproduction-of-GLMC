from enums import MatrixType
import torch
from torch import nn
from einops import rearrange
import copy
from scipy.optimize import linear_sum_assignment

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, bias=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # 3) normalized vector (apply c only when projecting back)
        norm_x = x / rms

        # 4) final output (+bias)
        out = self.weight * norm_x
        if self.bias is not None:
            out = out + self.bias

        return out


def absorb_ln_scale(model: nn.Module):
    with torch.no_grad():
        for block in model.transformer.h:
            block.ln_1.bias.copy_(block.ln_1.bias/block.ln_1.weight)
            block.attn.c_attn.weight.copy_(block.attn.c_attn.weight * block.ln_1.weight.unsqueeze(1))
            block.ln_1.weight.copy_(torch.ones(block.ln_1.weight.shape))

            block.ln_2.bias.copy_(block.ln_2.bias/block.ln_2.weight)
            block.mlp.c_fc.weight.copy_(block.mlp.c_fc.weight * block.ln_2.weight.unsqueeze(1))
            block.ln_2.weight.copy_(torch.ones(block.ln_2.weight.shape))
        
        with torch.no_grad():
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
        model.transformer.ln_f.bias.copy_(model.transformer.ln_f.bias/model.transformer.ln_f.weight)
        model.lm_head.weight.copy_(model.lm_head.weight * model.transformer.ln_f.weight)
        model.transformer.ln_f.weight.copy_(torch.ones(model.transformer.ln_f.weight.shape))


def replace_layernorm(module: nn.Module):
    with torch.no_grad():
        for name, child in module.named_children():
            # If the child itself contains other modules, replace them recursively.
            replace_layernorm(child)
            # If the child is an instance of LayerNorm, replace it.
            if isinstance(child, nn.LayerNorm):
                # Replace the module in its parent using setattr.
                rms_norm = RMSNorm(child.normalized_shape, eps=child.eps, bias=True)
                nn.init.ones_(rms_norm.weight)
                rms_norm.bias.copy_(child.bias)
                setattr(module, name, rms_norm)

def apply_mean_subtraction_to_weights(model: nn.Module):
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

def permute_mlp(model: nn.Module, idx: int, P: torch.Tensor):
    with torch.no_grad():
        model.transformer.h[idx].mlp.c_fc.weight.copy_(model.transformer.h[idx].mlp.c_fc.weight @ P)
        model.transformer.h[idx].mlp.c_fc.bias.copy_(model.transformer.h[idx].mlp.c_fc.bias @ P)

        model.transformer.h[idx].mlp.c_proj.weight.copy_(P.t() @ model.transformer.h[idx].mlp.c_proj.weight)

def make_Q(M: int, N: int) -> torch.Tensor:
    A = torch.randn(M, N)
    Q, R = torch.linalg.qr(A, mode="reduced")
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q            

def expand(model: nn.Module, n_embd_new: int):
    N_old = model.transformer.wte.weight.shape[1]
    M_new = n_embd_new
    assert N_old < M_new

    c = (N_old / M_new) ** 0.5
    O = make_Q(M=M_new, N=N_old).t()   # [N, M]

    trans_scaled_O = O.t() * c         # = Q * c

    with torch.no_grad():

        model.transformer.wte.weight.data = model.transformer.wte.weight @ O  
        model.transformer.wpe.weight.data = model.transformer.wpe.weight @ O

        for block in model.transformer.h:

            block.ln_1.bias.data   = (O.t() @ block.ln_1.bias) * (1.0 / c)
            block.ln_1.weight.data = torch.ones(M_new, dtype=block.ln_1.weight.dtype, device=block.ln_1.weight.device)
            block.ln_1.eps        *= (N_old / M_new)


            block.attn.c_attn.weight.data = trans_scaled_O @ block.attn.c_attn.weight

            block.attn.c_proj.weight.data = block.attn.c_proj.weight @ O
            block.attn.c_proj.bias.data   = block.attn.c_proj.bias @ O
            block.attn.c_proj.nf          = M_new

            block.ln_2.bias.data   = (O.t() @ block.ln_2.bias) * (1.0 / c)
            block.ln_2.weight.data = torch.ones(M_new, dtype=block.ln_2.weight.dtype, device=block.ln_2.weight.device)
            block.ln_2.eps        *= (N_old / M_new)

            block.mlp.c_fc.weight.data   = trans_scaled_O @ block.mlp.c_fc.weight
            block.mlp.c_proj.weight.data = block.mlp.c_proj.weight @ O
            block.mlp.c_proj.bias.data   = block.mlp.c_proj.bias @ O
            block.mlp.c_proj.nf          = M_new

            block.attn.embed_dim = M_new

        model.transformer.ln_f.bias.data   = (O.t() @ model.transformer.ln_f.bias) * (1.0 / c)
        model.transformer.ln_f.weight.data = torch.ones(M_new, dtype=model.transformer.ln_f.weight.dtype, device=model.transformer.ln_f.weight.device)
        model.transformer.ln_f.eps        *= (N_old / M_new)

        model.lm_head.weight.data = model.lm_head.weight @ O * c

    return model

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

        Q,K,V = c_attn_weight.data.chunk(3, dim = 0)

        Q = rearrange(Q, '(h d) m -> h d m', h = num_heads, m = embed_dim+1)
        K = rearrange(K, '(h d) m -> h d m', h = num_heads, m = embed_dim+1)
        V = rearrange(V, '(h d) m -> h d m', h = num_heads, m = embed_dim+1)

        Q = permute(Q, P)
        K = permute(K, P)
        V = permute(V, P)

        OUT = rearrange(c_proj.weight.data.t(), ' m (h d) -> m h d', h = num_heads, m = embed_dim)
        OUT = OUT.permute(1, 2, 0)

        OUT = permute(OUT, P)

        QK = torch.bmm(Q.transpose(1, 2), K)
        OUTV = OUT.transpose(1,2) @ V

        head_dim = Q.shape[1]

        Q_new = torch.zeros(QK.shape)
        K_new = torch.zeros(QK.shape)
        V_new = torch.zeros(QK.shape)

        OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1])

        for head_i in range(QK.size(0)):  # Loop through each batch
            def split(A):
                return A, torch.eye(A.shape[1])
            
            U_S_r, V_r = split(QK[head_i])
            Q_new[head_i] = U_S_r.t()
            K_new[head_i] = V_r

            U_S_r, V_r = split(OUTV[head_i])
            OUT_new[head_i] = U_S_r.t()
            V_new[head_i] = V_r

        Q_new = Q_new.reshape(-1, embed_dim+1)
        K_new = K_new.reshape(-1, embed_dim+1)
        V_new = V_new.reshape(-1, embed_dim+1)

        c_attn.weight.data = torch.cat((Q_new,K_new,V_new), dim=0)[:,:-1].t()
        c_attn.bias.data = torch.cat((Q_new,K_new,V_new), dim=0)[:,-1:].t()

        OUT_new = OUT_new.permute(2, 0, 1)
        OUT_new = OUT_new.reshape(embed_dim, -1)
        OUT_new = OUT_new.t()

        c_proj.weight.data = OUT_new

        # For c_attn
        model.transformer.h[layer_idx].attn.c_attn.nx = c_attn.weight.shape[0]  # input dim
        model.transformer.h[layer_idx].attn.c_attn.nf = c_attn.weight.shape[1]  # output dim
        model.transformer.h[layer_idx].attn.c_attn.weight = torch.nn.Parameter(c_attn.weight.clone())
        model.transformer.h[layer_idx].attn.c_attn.bias = torch.nn.Parameter(c_attn.bias.clone().squeeze())

        # For c_proj
        model.transformer.h[layer_idx].attn.c_proj.nx = c_proj.weight.shape[0]
        model.transformer.h[layer_idx].attn.c_proj.nf = c_proj.weight.shape[1]
        model.transformer.h[layer_idx].attn.c_proj.weight = torch.nn.Parameter(c_proj.weight.clone())

        model.transformer.h[layer_idx].attn.split_size = c_attn.weight.shape[1] // 3
        model.transformer.h[layer_idx].attn.head_dim = model.transformer.h[layer_idx].attn.embed_dim+1

def project_to_attn_circuits(model, layer_idx):
    with torch.no_grad():
        attn = copy.deepcopy(model.transformer.h[layer_idx].attn)
        num_heads = attn.num_heads
        embed_dim = attn.embed_dim

        c_attn = attn.c_attn
        c_proj = attn.c_proj

        c_attn_weight = torch.cat((c_attn.weight.t(), c_attn.bias.data.t().unsqueeze(1)), dim=1)

        Q,K,V = c_attn_weight.data.chunk(3, dim = 0)

        Q = rearrange(Q, '(h d) m -> h d m', h = num_heads, m = embed_dim+1)
        K = rearrange(K, '(h d) m -> h d m', h = num_heads, m = embed_dim+1)
        V = rearrange(V, '(h d) m -> h d m', h = num_heads, m = embed_dim+1)

        OUT = rearrange(c_proj.weight.data.t(), ' m (h d) -> m h d', h = num_heads, m = embed_dim)
        OUT = OUT.permute(1, 2, 0)

        QK = torch.bmm(Q.transpose(1, 2), K)
        OUTV = OUT.transpose(1,2) @ V

        Q_new = torch.zeros(QK.shape)
        K_new = torch.zeros(QK.shape)
        V_new = torch.zeros(QK.shape)

        OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1])

        for head_i in range(QK.size(0)):
            def split(A):
                return A, torch.eye(A.shape[1])
            
            U_S_r, V_r = split(QK[head_i])
            Q_new[head_i] = U_S_r.t()*(U_S_r.shape[1]**0.5/V.shape[1]**0.5)
            K_new[head_i] = V_r

            U_S_r, V_r = split(OUTV[head_i])
            OUT_new[head_i] = U_S_r.t()
            V_new[head_i] = V_r

        Q_new = Q_new.reshape(-1, embed_dim+1)
        K_new = K_new.reshape(-1, embed_dim+1)
        V_new = V_new.reshape(-1, embed_dim+1)


        c_attn.weight.data = torch.cat((Q_new,K_new,V_new), dim=0)[:,:-1].t()
        c_attn.bias.data = torch.cat((Q_new,K_new,V_new), dim=0)[:,-1:].t()

        OUT_new = OUT_new.permute(2, 0, 1)
        OUT_new = OUT_new.reshape(embed_dim, -1)
        OUT_new = OUT_new.t()

        c_proj.weight.data = OUT_new

        # For c_attn
        model.transformer.h[layer_idx].attn.c_attn.nx = c_attn.weight.shape[0]  # input dim
        model.transformer.h[layer_idx].attn.c_attn.nf = c_attn.weight.shape[1]  # output dim
        model.transformer.h[layer_idx].attn.c_attn.weight = torch.nn.Parameter(c_attn.weight.clone())
        model.transformer.h[layer_idx].attn.c_attn.bias = torch.nn.Parameter(c_attn.bias.clone().squeeze())

        # For c_proj
        model.transformer.h[layer_idx].attn.c_proj.nx = c_proj.weight.shape[0]
        model.transformer.h[layer_idx].attn.c_proj.nf = c_proj.weight.shape[1]
        model.transformer.h[layer_idx].attn.c_proj.weight = torch.nn.Parameter(c_proj.weight.clone())

        model.transformer.h[layer_idx].attn.split_size = c_attn.weight.shape[1] // 3
        model.transformer.h[layer_idx].attn.head_dim = model.transformer.h[layer_idx].attn.embed_dim+1

def _make_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """
    Project A to orthogonal matrix using SVD (differentiable).
    
    Args:
        A (torch.Tensor): Input matrix of shape (n, n).
        
    Returns:
        torch.Tensor: Closest orthogonal matrix to A.
    """
    device = A.device
    A = A.to("cpu")
    U, _, Vt = torch.linalg.svd(A)
    A = A.to(device)
    return torch.mm(U.to(device), Vt.to(device))

def _make_permutation(P):
    row_ind, col_ind = linear_sum_assignment(-P.detach().cpu().numpy())
    P = P * 0
    P[row_ind, col_ind] = 1
    return P


def project(A: torch.Tensor, matrix_type: MatrixType) -> torch.Tensor:
    """
    Project the input tensor `A` onto a specified class of matrices.

    This function takes an input tensor and projects it to the nearest matrix 
    of the specified type (permutation, soft-permutation, or orthogonal).
    """
    if matrix_type == MatrixType.PERM:
        return _make_permutation(A).detach() + (A - A.detach())
    elif matrix_type == MatrixType.SOFT_PERM:
        pass  # Implement the logic for SOFT_PERM if needed
    elif matrix_type == MatrixType.ORTHO:
        return _make_orthogonal(A)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

def interpolate(W0, W1, coeff):
    return coeff * W0 + (1-coeff) * W1

