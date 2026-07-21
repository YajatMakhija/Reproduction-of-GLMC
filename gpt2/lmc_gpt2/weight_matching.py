import copy
import torch 
from torch import nn
from einops import rearrange
import ot
from utils import permute_heads, permute_mlp


def ortho_residual(model: nn.Module, O: torch.Tensor):
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
    
    O = U @ Vh

    return O

def get_cost_heads(t0, t1, heads):
    cost_matrix = torch.zeros((heads, heads))
    for i in range(heads):
        for j in range(heads):
            diff = t0[i] - t1[j]
            cost_matrix[i, j] = torch.sqrt(torch.sum(diff ** 2))  # Frobenius norm
    return cost_matrix

def otify(cost):
    ot_map = ot.emd(
            torch.ones(cost.shape[0]) / cost.shape[0], torch.ones(cost.shape[0]) / cost.shape[0], cost
        )
    return ot_map*cost.shape[0]

def _ot_cost_matrix(X, Y, metric="euclidean2", eps=1e-8):
    # X,Y: [N, M] (we match columns)
    if metric == "euclidean2":
        X2 = (X**2).sum(dim=0, keepdim=True)         # [1,M]
        Y2 = (Y**2).sum(dim=0, keepdim=True)         # [1,M]
        C = X2.T + Y2 - 2 * (X.T @ Y)                # [M,M]
        return torch.clamp(C, min=0)
    elif metric == "cosine":
        Xn = X / (X.norm(dim=0, keepdim=True) + eps)
        Yn = Y / (Y.norm(dim=0, keepdim=True) + eps)
        S = Xn.T @ Yn
        return 1.0 - S                               # minimize 1 - cosine
    else:
        raise ValueError("metric must be 'euclidean2' or 'cosine'")

def compute_optimal_permutation_matrix_ot(t1, t2, metric="euclidean2", use_sinkhorn=False, reg=0.01):
    assert t1.shape == t2.shape and t1.dim() == 2
    N, M = t1.shape

    C_t = _ot_cost_matrix(t1, t2, metric=metric).detach()
    C = C_t.cpu().numpy()
    a = ot.unif(M)  
    b = ot.unif(M)   

    if use_sinkhorn:
        T = ot.sinkhorn(a, b, C, reg)
        T_torch = torch.from_numpy(T).to(t1.device, dtype=t1.dtype)
        idx = T_torch.argmax(dim=1)             
        P = torch.zeros((M, M), device=t1.device, dtype=t1.dtype)
        P[torch.arange(M, device=t1.device), idx] = 1.0
        if P.sum().item() != M or (P.sum(dim=0) > 1.0 + 1e-6).any():
            T_emd = ot.emd(a, b, C)               
            P = torch.from_numpy(T_emd).to(t1.device, dtype=t1.dtype)
            P = (P * M).round()                    
    else:
        T = ot.emd(a, b, C)                    
        P = torch.from_numpy(T).to(t1.device, dtype=t1.dtype)
        P = (P * M).round()                    

    t2_aligned = t2 @ P
    frobenius_norm = torch.norm(t1 - t2_aligned, p='fro').item()
    mean_cosine = torch.nn.functional.cosine_similarity(t1, t2_aligned, dim=0).mean().item()

    perm_idx = P.argmax(dim=1)               

    return P, frobenius_norm, mean_cosine, perm_idx

def weight_matching(
    model0,
    model1,
    heads,
    iterations=15,
    permutations_only=False,
    token_freqs=None,       
    include_always=None,     
    block_size=None,        
):
    device = next(model0.parameters()).device

    active_token_ids = None
    if token_freqs is not None:
        active_token_ids = (token_freqs > 0).nonzero(as_tuple=False).flatten()
        if include_always:
            extra = torch.tensor(include_always, dtype=torch.long)
            active_token_ids = torch.unique(torch.cat([active_token_ids, extra]))
        active_token_ids = active_token_ids.to(device)

    for i in range(iterations):
        tok0 = model0.transformer.wte.weight.data 
        tok1 = model1.transformer.wte.weight.data  
        if active_token_ids is not None:
            tok0 = tok0.index_select(0, active_token_ids)  
            tok1 = tok1.index_select(0, active_token_ids)   

        pos0 = model0.transformer.wpe.weight.data        
        pos1 = model1.transformer.wpe.weight.data        
        if block_size is not None:
            pos0 = pos0[:block_size]                      
            pos1 = pos1[:block_size]                       

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

        layers_0 = [layer / layer.shape[1]**0.5 for layer in layers_0]
        layers_1 = [layer / layer.shape[1]**0.5 for layer in layers_1]

        if permutations_only:
            O, _, _, _ = compute_optimal_permutation_matrix_ot(torch.cat(layers_0, dim=1).t(), torch.cat(layers_1, dim=1).t(), metric="euclidean2", use_sinkhorn=False)
            O = O.t() 
        else:
            O = compute_optimal_orthogonal_matrix(torch.cat(layers_0, dim=1).t(), torch.cat(layers_1, dim=1).t())

        ortho_residual(model1, O)

        for layer_i in range(len(model1.transformer.h)):
            def get_qkv(model):
                attn = copy.deepcopy(model.transformer.h[layer_i].attn)
                num_heads = attn.num_heads
                embed_dim = attn.embed_dim

                c_attn = attn.c_attn
                c_proj = attn.c_proj

                c_attn_weight = torch.cat((c_attn.weight.t(), c_attn.bias.data.t().unsqueeze(1)), dim=1)

                Q, K, V = c_attn_weight.data.chunk(3, dim=0)

                Q = rearrange(Q, '(h d) m -> h d m', h=num_heads, m=embed_dim+1)
                K = rearrange(K, '(h d) m -> h d m', h=num_heads, m=embed_dim+1)
                V = rearrange(V, '(h d) m -> h d m', h=num_heads, m=embed_dim+1)
                OUT = rearrange(c_proj.weight.data.t(), ' m (h d) -> m h d', h=num_heads, m=embed_dim)
                OUT = OUT.permute(1, 2, 0)

                QK = torch.bmm(Q.transpose(1, 2), K)
                OUTV = OUT.transpose(1, 2) @ V
                return QK, OUTV

            QK0, OUTV0 = get_qkv(model0)
            QK1, OUTV1 = get_qkv(model1)
            cost_qk = get_cost_heads(QK0, QK1, heads=heads)
            cost_outv = get_cost_heads(OUTV0, OUTV1, heads=heads)
            cost = cost_qk + cost_outv
            P = otify(cost).to(QK0.device)

            permute_heads(model1, layer_i, P)

            ff0 = torch.cat((
                model0.transformer.h[layer_i].mlp.c_fc.weight.data.t(),
                model0.transformer.h[layer_i].mlp.c_fc.bias.unsqueeze(1),
                model0.transformer.h[layer_i].mlp.c_proj.weight.data), dim=1)

            ff1 = torch.cat((
                model1.transformer.h[layer_i].mlp.c_fc.weight.data.t(),
                model1.transformer.h[layer_i].mlp.c_fc.bias.unsqueeze(1),
                model1.transformer.h[layer_i].mlp.c_proj.weight.data), dim=1)

            cost_ff = torch.cdist(
                ff0 / torch.norm(ff0, dim=-1, keepdim=True),
                ff1 / torch.norm(ff1, dim=-1, keepdim=True),
                p=1).cpu()

            P_ff = otify(cost_ff).to(ff0.device)

            permute_mlp(model1, layer_i, P=P_ff.t())

    return model1

