"""Depth-heterogeneous GPT-2 merging on Tiny Shakespeare (GLMC reproduction + extension)."""

import copy
import gc
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


SMOKE = True
USE_BF16 = True
SEED_DEEP, SEED_SHALLOW = 0, 1
CKPT_DEEP = None
CKPT_SHALLOW = None
SKIP_TRAIN = False

SEQ_LEN = 512
BATCH_SIZE = 64
EVAL_BATCH = 32
EPOCHS = 100
WARMUP_STEPS = 100
WM_ITERS = 15
ACT_BATCHES = 20
ACT_MAX_TOKENS = 8192
LEARNED_EPOCHS = 2
LEARNED_BATCHES = 200

OUTPUT_ROOT = Path("outputs/shakespeare_depth_het_marimo")
PARK = OUTPUT_ROOT / "park"
PLOTS = OUTPUT_ROOT / "plots"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if SMOKE:
    EPOCHS = 1
    WARMUP_STEPS = 1
    WM_ITERS = 2
    ACT_BATCHES = 2
    ACT_MAX_TOKENS = 1024
    LEARNED_EPOCHS = 1
    LEARNED_BATCHES = 2

MOCK_TRAINING = bool(SMOKE)

_gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
_vram = (
    torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if DEVICE.type == "cuda" else 0.0
)


if DEVICE.type != "cuda":
    if not SMOKE:
        raise RuntimeError("CUDA GPU required for the full run. Enable GPU in the runtime.")
    print("[gpu] WARNING: no CUDA — SMOKE pass will run on CPU", flush=True)
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _free, _total = torch.cuda.mem_get_info()
    print(
        f"[gpu] {torch.cuda.get_device_name(0)}  "
        f"free={_free/1e9:.1f}/{_total/1e9:.1f} GB  torch={torch.__version__}",
        flush=True,
    )
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PARK.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


VOCAB_SIZE = 50257

@dataclass
class TrainCfg:


    dim: int = 512
    heads: int = 8
    mlp_dim: int = 2048
    seq_len: int = 512
    dropout: float = 0.1
    deep_depth: int = 12
    shallow_depth: int = 6
    batch_size: int = 64
    epochs: int = 100
    lr: float = 2.5e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    use_bf16: bool = True
    wm_iterations: int = 15
    act_batches: int = 20
    learned_epochs: int = 2
    learned_batches_per_epoch: int = 200
    narrow_lambda_lo: float = 0.4
    narrow_lambda_hi: float = 0.6

CFG = TrainCfg(
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    use_bf16=bool(USE_BF16),
    wm_iterations=WM_ITERS,
    act_batches=ACT_BATCHES,
    learned_epochs=LEARNED_EPOCHS,
    learned_batches_per_epoch=LEARNED_BATCHES,
)
print("[cfg]", CFG, " MOCK=", MOCK_TRAINING, flush=True)


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm: RMSNorm(Ox) = O * RMSNorm(x) for orthogonal O.
    Replaces LayerNorm everywhere (Stage A): no mean subtraction, no affine."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dim, self.dim_head = heads, dim, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))

class GPTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.is_identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return x
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

    @torch.no_grad()
    def make_identity(self):
        """Silent block: bypass Attn/MLP residuals AND zero their output
        maps, so state-dict interpolation semantics stay exact."""
        self.is_identity = True
        nn.init.zeros_(self.attn.qkv.weight)
        nn.init.zeros_(self.attn.qkv.bias)
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.attn.proj.bias)
        nn.init.zeros_(self.mlp.fc1.weight)
        nn.init.zeros_(self.mlp.fc1.bias)
        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)

class RMSGPT(nn.Module):
    """GPT-2 style causal LM with parameter-free RMSNorm and TIED
    input/output embeddings (tying is symmetry-consistent: both the
    embedding writer and the lm-head reader transform as E <- E @ O^T)."""

    def __init__(self, *, vocab=VOCAB_SIZE, dim=256, depth=12, heads=4,
                 mlp_dim=1024, seq_len=256, dropout=0.1):
        super().__init__()
        self.dim, self.depth, self.heads, self.mlp_dim = dim, depth, heads, mlp_dim
        self.seq_len = seq_len
        self.wte = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.final_norm = RMSNorm(dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.pos_emb(pos).unsqueeze(0))
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        return F.linear(x, self.wte.weight)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

def build_gpt(depth: int, cfg: TrainCfg = CFG) -> RMSGPT:
    return RMSGPT(
        vocab=VOCAB_SIZE, dim=cfg.dim, depth=depth, heads=cfg.heads,
        mlp_dim=cfg.mlp_dim, seq_len=cfg.seq_len, dropout=cfg.dropout,
    )


SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

def _load_tokens(cache_dir: Path) -> torch.Tensor:
    tok_path = cache_dir / "tokens_gpt2bpe.pt"
    if tok_path.exists():
        return torch.load(tok_path, weights_only=False)
    import urllib.request
    import tiktoken
    text = urllib.request.urlopen(SHAKESPEARE_URL, timeout=60).read().decode("utf-8")
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(text), dtype=torch.long)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, tok_path)
    print(f"[data] tiny shakespeare: {len(text):,} chars -> {len(tokens):,} BPE tokens")
    return tokens

class GPUTokenLoader:
    """DataLoader stand-in over GPU-resident (x, y) sequence chunks."""

    def __init__(self, x, y, batch_size, shuffle, drop_last=False):
        self.x, self.y = x, y
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def __len__(self):
        n = self.x.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = self.x.shape[0]
        dev = self.x.device
        order = (torch.randperm(n, device=dev) if self.shuffle
                 else torch.arange(n, device=dev))
        end = n - (n % self.batch_size) if self.drop_last else n
        for s in range(0, end, self.batch_size):
            idx = order[s:s + self.batch_size]
            yield self.x[idx], self.y[idx]

def get_token_loaders(device, batch_size=32, eval_batch=64, seq_len=256,
                      cache_dir="shakespeare_cache"):
    """Returns (train_loader, stat_loader [same data, for activation stats],
    val_loader). 90/10 contiguous train/val split of the token stream."""
    cache = Path(cache_dir)
    try:
        tokens = _load_tokens(cache)
    except Exception as e:
        if not MOCK_TRAINING:
            raise
        print(f"[data] download unavailable ({e}); synthetic MOCK tokens")
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, VOCAB_SIZE, (120_000,), generator=g)

    n_train = int(0.9 * len(tokens))
    splits = {"train": tokens[:n_train], "val": tokens[n_train:]}
    chunks = {}
    for name, t in splits.items():
        n = (len(t) - 1) // seq_len
        x = t[: n * seq_len].view(n, seq_len)
        y = t[1: n * seq_len + 1].view(n, seq_len)
        chunks[name] = (x.to(device), y.to(device))
        print(f"[data] {name}: {len(t):,} tokens -> {n} chunks of {seq_len}")

    train_ld = GPUTokenLoader(*chunks["train"], batch_size, shuffle=True, drop_last=True)
    stat_ld = GPUTokenLoader(*chunks["train"], batch_size, shuffle=True)
    val_ld = GPUTokenLoader(*chunks["val"], eval_batch, shuffle=False)
    return train_ld, stat_ld, val_ld


@torch.no_grad()
def evaluate_lm_loss(model, loader, device, max_batches: Optional[int] = None):
    """Mean autoregressive cross-entropy per token on the loader."""
    model.eval()
    loss_sum = 0.0
    tok_sum = 0
    use_amp = device.type == "cuda" and CFG.use_bf16
    for i, (x, y) in enumerate(loader):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss_sum += loss.item() * y.numel()
        tok_sum += y.numel()
        if max_batches is not None and i + 1 >= max_batches:
            break
        if MOCK_TRAINING:
            break
    return loss_sum / max(tok_sum, 1)

def _lr_at(step, total_steps, warmup, base_lr):
    """Linear warmup then cosine decay to 0 (per-step, trivially resumable)."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

def train_one(model, train_ld, val_ld, device, seed, tag, cfg=CFG) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda" and cfg.use_bf16
    steps_per_epoch = max(1 if MOCK_TRAINING else len(train_ld), 1)
    total_steps = cfg.epochs * steps_per_epoch


    out = Path(OUTPUT_ROOT) / "models" / f"{tag}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"
    resume_path = out / "resume.pt"
    log_path = out / "history.json"
    print(f"[train] {tag} depth={model.depth} params={model.count_params()/1e6:.2f}M -> {out}")

    history = []
    start_ep = 0
    if ckpt_path.exists() and log_path.exists():
        history = json.loads(log_path.read_text())
        start_ep = min(len(history), cfg.epochs)
        if start_ep > 0:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
            if resume_path.exists():
                rs = torch.load(resume_path, map_location=device, weights_only=False)
                opt.load_state_dict(rs["opt"])
            print(f"[train] {tag}: resumed at epoch {start_ep}/{cfg.epochs}")

    vloss = history[-1].get("val_loss", 0.0) if history else 0.0
    gstep = start_ep * steps_per_epoch
    for ep in range(start_ep, cfg.epochs):
        model.train()
        run_loss, run_tok = 0.0, 0
        for x, y in tqdm(train_ld, desc=f"{tag} ep{ep+1}/{cfg.epochs}", leave=False):
            lr = _lr_at(gstep, total_steps, cfg.warmup_steps, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            gstep += 1
            run_loss += loss.item() * y.numel()
            run_tok += y.numel()
            if MOCK_TRAINING:
                break
        max_b = 2 if MOCK_TRAINING else None
        vloss = evaluate_lm_loss(model, val_ld, device, max_batches=max_b)
        history.append({"epoch": ep + 1, "train_loss": run_loss / max(run_tok, 1),
                        "val_loss": vloss, "val_ppl": math.exp(min(vloss, 20.0)),
                        "lr": lr})
        log_path.write_text(json.dumps(history, indent=2))
        torch.save({
            "state_dict": model.state_dict(), "depth": model.depth,
            "cfg": asdict(cfg), "tag": tag, "seed": seed, "val_loss": vloss,
        }, ckpt_path)
        torch.save({"opt": opt.state_dict(), "epoch": ep + 1}, resume_path)
        print(f"[train] {tag} ep{ep+1}: val_loss={vloss:.4f}  "
              f"ppl={math.exp(min(vloss, 20.0)):.1f}  lr={lr:.2e}")
    return out


@torch.no_grad()
def expand_shallow_gpt2(shallow: RMSGPT, target_depth: int = 12) -> RMSGPT:
    """Map 6 trained layers -> even indices {0,2,...,10}; odd slots = identity."""
    assert shallow.depth * 2 == target_depth
    out = build_gpt(target_depth, CFG)
    out.wte.load_state_dict(shallow.wte.state_dict())
    out.pos_emb.load_state_dict(shallow.pos_emb.state_dict())

    for s_i, blk in enumerate(shallow.blocks):
        d_i = 2 * s_i
        out.blocks[d_i].load_state_dict(blk.state_dict())
        out.blocks[d_i].is_identity = False

    for d_i in range(1, target_depth, 2):
        out.blocks[d_i].make_identity()


    x = torch.randn(2, 16, CFG.dim)
    for d_i in range(1, target_depth, 2):
        assert torch.allclose(x, out.blocks[d_i](x)), f"block {d_i} not identity"

    print(f"[expand] {shallow.depth} -> {target_depth} (even=trained, odd=identity) "
          f"params={out.count_params()/1e6:.2f}M")
    return out


def hungarian_perm(cost: torch.Tensor) -> torch.Tensor:
    """Minimize sum_i cost[i, pi(i)] -> hard permutation matrix."""
    r, c = linear_sum_assignment(cost.detach().float().cpu().numpy())
    P = torch.zeros_like(cost)
    P[r, c] = 1.0
    return P

def ste_perm(A: torch.Tensor) -> torch.Tensor:
    """STE: forward = Hungarian(-A), backward = identity through A."""
    P = hungarian_perm(-A)
    return P.detach() + (A - A.detach())

def project_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """O = U V^T closest orthogonal matrix (Procrustes / SVD)."""
    U, _, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    return (U @ Vh).to(dtype=A.dtype)

def corr_cost(X0: torch.Tensor, X1: torch.Tensor) -> torch.Tensor:
    X0 = (X0 - X0.mean(0)) / (X0.std(0) + 1e-8)
    X1 = (X1 - X1.mean(0)) / (X1.std(0) + 1e-8)
    return -(X0.T @ X1) / max(X0.shape[0], 1)

@torch.no_grad()
def apply_residual_O(model: RMSGPT, O: torch.Tensor):
    """Writers: W <- O @ W (or E <- E @ O^T for embeddings); readers: W <- W @ O^T.
    Tied lm head follows wte automatically."""
    model.wte.weight.data = model.wte.weight.data @ O.T
    model.pos_emb.weight.data = model.pos_emb.weight.data @ O.T

    for blk in model.blocks:
        if blk.is_identity:
            continue
        blk.attn.qkv.weight.data = blk.attn.qkv.weight.data @ O.T
        blk.attn.proj.weight.data = O @ blk.attn.proj.weight.data
        blk.attn.proj.bias.data = O @ blk.attn.proj.bias.data
        blk.mlp.fc1.weight.data = blk.mlp.fc1.weight.data @ O.T
        blk.mlp.fc2.weight.data = O @ blk.mlp.fc2.weight.data
        blk.mlp.fc2.bias.data = O @ blk.mlp.fc2.bias.data

@torch.no_grad()
def permute_mlp_channels(blk: GPTBlock, P: torch.Tensor):
    if blk.is_identity:
        return
    blk.mlp.fc1.weight.data = P @ blk.mlp.fc1.weight.data
    blk.mlp.fc1.bias.data = P @ blk.mlp.fc1.bias.data
    blk.mlp.fc2.weight.data = blk.mlp.fc2.weight.data @ P.T

def _perm_heads_qkv(W, P, H, Dh, D):

    t = W.view(3, H, Dh, D)
    t = torch.einsum("hk,ckdD->chdD", P, t)
    return t.reshape(3 * H * Dh, D)

def _perm_heads_qkv_bias(b, P, H, Dh):

    t = b.view(3, H, Dh)
    t = torch.einsum("hk,ckd->chd", P, t)
    return t.reshape(3 * H * Dh)

def _perm_heads_out(W, P, H, Dh, D):

    t = W.view(D, H, Dh).permute(1, 0, 2).reshape(H, D * Dh)
    t = P @ t
    return t.view(H, D, Dh).permute(1, 0, 2).contiguous().view(D, H * Dh)

@torch.no_grad()
def permute_attention_heads(blk: GPTBlock, P_heads: torch.Tensor):
    if blk.is_identity:
        return
    H, Dh, D = blk.attn.heads, blk.attn.dim_head, blk.attn.dim
    P = P_heads.to(dtype=blk.attn.qkv.weight.dtype)
    blk.attn.qkv.weight.data = _perm_heads_qkv(blk.attn.qkv.weight.data, P, H, Dh, D)
    blk.attn.qkv.bias.data = _perm_heads_qkv_bias(blk.attn.qkv.bias.data, P, H, Dh)
    blk.attn.proj.weight.data = _perm_heads_out(blk.attn.proj.weight.data, P, H, Dh, D)


@torch.no_grad()
def _collect_acts(model: RMSGPT, loader, device, max_batches: int, max_tokens: int = 8192):
    """Capped activation banks: MLP hidden per active block, per-head summary,
    residual stream at the network output boundary."""
    model.eval()
    n = model.depth
    mlp_acts = [[] for _ in range(n)]
    head_acts = [[] for _ in range(n)]
    residual = []
    hooks = []
    tok_left = [max_tokens for _ in range(n)]

    for i, blk in enumerate(model.blocks):
        if blk.is_identity:
            continue

        def _make_hook(idx):
            def fn(_m, _inp, out):
                if tok_left[idx] <= 0:
                    return
                flat = out.detach().reshape(-1, out.shape[-1])
                take = min(flat.size(0), tok_left[idx])
                if flat.size(0) > take:
                    idx_r = torch.randperm(flat.size(0), device=flat.device)[:take]
                    flat = flat[idx_r]
                mlp_acts[idx].append(flat.float())
                tok_left[idx] -= flat.size(0)
            return fn

        hooks.append(blk.mlp.fc1.register_forward_hook(_make_hook(i)))

    for bi, (idx_b, _) in enumerate(loader):
        if bi >= max_batches:
            break
        B, T = idx_b.shape
        pos = torch.arange(T, device=device)
        x = model.wte(idx_b) + model.pos_emb(pos).unsqueeze(0)
        for i, blk in enumerate(model.blocks):
            if blk.is_identity:
                x = blk(x)
                continue
            h = blk.norm1(x)
            q, k, v = blk.attn.qkv(h).chunk(3, dim=-1)
            q = q.view(B, T, blk.attn.heads, blk.attn.dim_head).transpose(1, 2)
            k = k.view(B, T, blk.attn.heads, blk.attn.dim_head).transpose(1, 2)
            v = v.view(B, T, blk.attn.heads, blk.attn.dim_head).transpose(1, 2)
            hout = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            head_acts[i].append(hout.abs().mean((2, 3)).float())
            x = blk(x)
        residual.append(x.mean(1).float())
        del idx_b, x
        if device.type == "cuda" and (bi % 2 == 1):
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    def _cat_cpu(chunks):
        if not chunks:
            return None
        t = torch.cat(chunks, 0)
        out = t.cpu()
        del t
        return out

    mlp = [_cat_cpu(a) if a else None for a in mlp_acts]
    heads = [_cat_cpu(a) if a else None for a in head_acts]
    res = _cat_cpu(residual)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mlp, heads, res

@torch.no_grad()
def activation_match(model_a, model_b, loader, device, cfg, max_tokens: int = 8192) -> RMSGPT:
    A, B = model_a.to(device).eval(), copy.deepcopy(model_b).to(device).eval()
    mlp_a, head_a, res_a = _collect_acts(A, loader, device, cfg.act_batches, max_tokens=max_tokens)
    mlp_b, head_b, res_b = _collect_acts(B, loader, device, cfg.act_batches, max_tokens=max_tokens)


    Ra, Rb = res_a.to(device), res_b.to(device)
    P_res = hungarian_perm(corr_cost(Ra, Rb))
    apply_residual_O(B, P_res)
    del mlp_b, head_b, res_a, res_b, Ra, Rb
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mlp_b, head_b, _ = _collect_acts(B, loader, device, cfg.act_batches, max_tokens=max_tokens)
    for i, blk in enumerate(B.blocks):
        if blk.is_identity or mlp_a[i] is None or mlp_b[i] is None:
            continue
        permute_mlp_channels(blk, hungarian_perm(corr_cost(mlp_a[i].to(device), mlp_b[i].to(device))))
        permute_attention_heads(blk, hungarian_perm(corr_cost(head_a[i].to(device), head_b[i].to(device))))

    del mlp_a, mlp_b, head_a, head_b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[method2] activation matching done")
    return B


@torch.no_grad()
def _paired_residual_stacks(model_a: RMSGPT, model_b: RMSGPT):
    """Equal-width residual stacks (D, *) using only layers active in both.
    Tied lm head == wte, so the embedding enters once."""
    cols_a = [model_a.wte.weight.data.T, model_a.pos_emb.weight.data.T]
    cols_b = [model_b.wte.weight.data.T, model_b.pos_emb.weight.data.T]
    for ba, bb in zip(model_a.blocks, model_b.blocks):
        if ba.is_identity or bb.is_identity:
            continue
        cols_a += [
            ba.attn.proj.weight.data, ba.mlp.fc2.weight.data,
            ba.attn.qkv.weight.data.T, ba.mlp.fc1.weight.data.T,
        ]
        cols_b += [
            bb.attn.proj.weight.data, bb.mlp.fc2.weight.data,
            bb.attn.qkv.weight.data.T, bb.mlp.fc1.weight.data.T,
        ]
    return torch.cat(cols_a, 1), torch.cat(cols_b, 1)

@torch.no_grad()
def weight_match(model_a: RMSGPT, model_b: RMSGPT, cfg: TrainCfg) -> RMSGPT:
    A, B = model_a.eval(), copy.deepcopy(model_b).eval()
    H, Dh, D = CFG.heads, CFG.dim // CFG.heads, CFG.dim

    for it in range(cfg.wm_iterations):
        Sa, Sb = _paired_residual_stacks(A, B)
        O = project_orthogonal(Sa @ Sb.T)
        apply_residual_O(B, O)

        for ba, bb in zip(A.blocks, B.blocks):
            if ba.is_identity or bb.is_identity:
                continue
            fa = torch.cat([ba.mlp.fc1.weight.data, ba.mlp.fc2.weight.data.T], 1)
            fb = torch.cat([bb.mlp.fc1.weight.data, bb.mlp.fc2.weight.data.T], 1)
            fa_n = fa / (fa.norm(1, keepdim=True) + 1e-8)
            fb_n = fb / (fb.norm(1, keepdim=True) + 1e-8)
            permute_mlp_channels(bb, hungarian_perm(torch.cdist(fa_n, fb_n, p=1)))

            def head_feats(blk):
                Wqkv = blk.attn.qkv.weight.data.view(3, H, Dh, D)
                Wout = blk.attn.proj.weight.data.view(D, H, Dh).permute(1, 2, 0)
                return torch.stack([
                    torch.cat([Wqkv[0, h].reshape(-1), Wqkv[1, h].reshape(-1),
                               Wqkv[2, h].reshape(-1), Wout[h].reshape(-1)])
                    for h in range(H)
                ])

            ha, hb = head_feats(ba), head_feats(bb)
            ha_n = ha / (ha.norm(1, keepdim=True) + 1e-8)
            hb_n = hb / (hb.norm(1, keepdim=True) + 1e-8)
            permute_attention_heads(bb, hungarian_perm(torch.cdist(ha_n, hb_n, p=1)))

        print(f"[method3] weight matching iter {it+1}/{cfg.wm_iterations}")
    return B


def sinkhorn(A: torch.Tensor, iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Soft permutation: exp(A) pushed toward the Birkhoff polytope by
    Sinkhorn-Knopp row/col normalization. Fully differentiable — used
    during training of the permutations-only variant (hard Hungarian STE
    is piecewise-constant and freezes at its warm start)."""
    Q = torch.exp(A - A.max())
    for _ in range(iters):
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        Q = Q / (Q.sum(dim=0, keepdim=True) + eps)
    return Q

class AlignmentState(nn.Module):
    def __init__(self, depth, dim, heads, mlp_dim, permutations_only: bool,
                 soft_perm: bool = False):
        super().__init__()
        self.permutations_only = permutations_only
        self.soft_perm = soft_perm

        scale = 4.6 if soft_perm else 1.0
        self.A_res = nn.Parameter(torch.eye(dim))
        self.A_heads = nn.ParameterList(
            [nn.Parameter(scale * torch.eye(heads)) for _ in range(depth)])
        self.A_mlp = nn.ParameterList(
            [nn.Parameter(scale * torch.eye(mlp_dim)) for _ in range(depth)])

    def projected(self, soft: bool = False):
        if self.permutations_only:
            O = torch.eye(self.A_res.shape[0], device=self.A_res.device, dtype=self.A_res.dtype)
        else:

            O_valid = project_orthogonal(self.A_res)
            O = O_valid.detach() + (self.A_res - self.A_res.detach())
        proj = sinkhorn if (soft and self.soft_perm) else ste_perm
        P_heads = [proj(a) for a in self.A_heads]
        P_mlp = [proj(a) for a in self.A_mlp]
        return O, P_heads, P_mlp

@torch.no_grad()
def hard_align(model_b: RMSGPT, O, P_heads, P_mlp) -> RMSGPT:
    B = copy.deepcopy(model_b)
    apply_residual_O(B, O.detach())
    for i, blk in enumerate(B.blocks):
        if blk.is_identity:
            continue
        permute_mlp_channels(blk, P_mlp[i].detach())
        permute_attention_heads(blk, P_heads[i].detach())
    return B

def ste_interpolated_logits(
    model_a: RMSGPT,
    model_b: RMSGPT,
    align: AlignmentState,
    idx: torch.Tensor,
    lam: float,
    soft: bool = False,
) -> torch.Tensor:
    """Blend weights with (soft/STE) P and SVD-projected O; keep graph into
    the align latents. W_merged = (1-lam) W_A + lam (O . P . W_B)."""
    O, P_heads, P_mlp = align.projected(soft=soft)
    if align.permutations_only:
        O = torch.eye(CFG.dim, device=idx.device, dtype=model_a.wte.weight.dtype)

    H, Dh, D = CFG.heads, CFG.dim // CFG.heads, CFG.dim
    B, T = idx.shape


    E = (1.0 - lam) * model_a.wte.weight + lam * (model_b.wte.weight @ O.T)
    Epos = (1.0 - lam) * model_a.pos_emb.weight + lam * (model_b.pos_emb.weight @ O.T)
    pos = torch.arange(T, device=idx.device)
    h = F.embedding(idx, E) + F.embedding(pos, Epos).unsqueeze(0)


    for i in range(model_a.depth):
        ba_blk, bb_blk = model_a.blocks[i], model_b.blocks[i]

        Wqkv_a, bqkv_a = ba_blk.attn.qkv.weight, ba_blk.attn.qkv.bias
        Wqkv_b = _perm_heads_qkv(bb_blk.attn.qkv.weight @ O.T, P_heads[i], H, Dh, D)
        bqkv_b = _perm_heads_qkv_bias(bb_blk.attn.qkv.bias, P_heads[i], H, Dh)
        Wqkv = (1.0 - lam) * Wqkv_a + lam * Wqkv_b
        bqkv = (1.0 - lam) * bqkv_a + lam * bqkv_b

        Wout_a = ba_blk.attn.proj.weight
        Wout_b = _perm_heads_out(O @ bb_blk.attn.proj.weight, P_heads[i], H, Dh, D)
        Wout = (1.0 - lam) * Wout_a + lam * Wout_b
        bout = (1.0 - lam) * ba_blk.attn.proj.bias + lam * (O @ bb_blk.attn.proj.bias)


        hn = ba_blk.norm1(h)
        q, k, v = F.linear(hn, Wqkv, bqkv).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        aout = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        aout = aout.transpose(1, 2).contiguous().view(B, T, D)
        h = h + F.linear(aout, Wout, bout)


        W1 = (1.0 - lam) * ba_blk.mlp.fc1.weight + lam * (P_mlp[i] @ (bb_blk.mlp.fc1.weight @ O.T))
        b1 = (1.0 - lam) * ba_blk.mlp.fc1.bias + lam * (P_mlp[i] @ bb_blk.mlp.fc1.bias)
        W2 = (1.0 - lam) * ba_blk.mlp.fc2.weight + lam * (O @ (bb_blk.mlp.fc2.weight @ P_mlp[i].T))
        b2 = (1.0 - lam) * ba_blk.mlp.fc2.bias + lam * (O @ bb_blk.mlp.fc2.bias)

        hn = ba_blk.norm2(h)
        h = h + F.linear(ba_blk.mlp.act(F.linear(hn, W1, b1)), W2, b2)

    h = model_a.final_norm(h)
    return F.linear(h, E)

def train_learned_matching(
    model_a, model_b_wm, train_ld, device, permutations_only: bool, cfg: TrainCfg,
) -> RMSGPT:
    align = AlignmentState(
        model_a.depth, CFG.dim, CFG.heads, CFG.mlp_dim, permutations_only,
        soft_perm=permutations_only,
    ).to(device)
    opt = torch.optim.Adam(align.parameters(), lr=1e-3)
    model_a, model_b_wm = model_a.to(device).eval(), model_b_wm.to(device).eval()
    for p in list(model_a.parameters()) + list(model_b_wm.parameters()):
        p.requires_grad_(False)

    tag = "perm-only" if permutations_only else "full(O+P)"
    print(f"[learned] {tag}  epochs={cfg.learned_epochs}")

    for ep in range(cfg.learned_epochs):
        total = nstep = 0
        for bi, (x, y) in enumerate(train_ld):
            if bi >= cfg.learned_batches_per_epoch:
                break
            lam = random.uniform(cfg.narrow_lambda_lo, cfg.narrow_lambda_hi)
            opt.zero_grad(set_to_none=True)
            logits = ste_interpolated_logits(model_a, model_b_wm, align, x, lam,
                                             soft=permutations_only)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            if not permutations_only:
                with torch.no_grad():
                    align.A_res.copy_(project_orthogonal(align.A_res))
            total += loss.item()
            nstep += 1
            if MOCK_TRAINING:
                break
        print(f"[learned] {tag} ep{ep+1}: loss={total / max(nstep, 1):.4f}")

    with torch.no_grad():

        O, Ph, Pm = align.projected(soft=False)
        if permutations_only:
            O = torch.eye(CFG.dim, device=device)
        moved = sum(int((p.argmax(1) != torch.arange(p.shape[0], device=p.device)).sum())
                    for p in Pm)
        print(f"[learned] {tag}: MLP assignments moved from identity: {moved}")
        return hard_align(model_b_wm, O, Ph, Pm)


@torch.no_grad()
def measure_barrier(model_a, aligned_b, val_ld, device, method_name: str) -> Dict:
    """
    Barrier = max_lambda [ Loss(lambda) - ((1-lambda) Loss(0) + lambda Loss(1)) ]
    with W(lambda) = (1-lambda) W_A + lambda W_B_aligned.
    """
    model_a, aligned_b = model_a.to(device).eval(), aligned_b.to(device).eval()
    sd_a = model_a.state_dict()
    sd_b = aligned_b.state_dict()
    shell = copy.deepcopy(model_a).to(device)
    coeffs = [round(0.1 * i, 1) for i in range(11)]
    losses = {}
    max_b = 2 if MOCK_TRAINING else None

    for lam in coeffs:
        merged = {}
        for k, va in sd_a.items():
            vb = sd_b[k]
            if torch.is_floating_point(va) and va.shape == vb.shape:
                merged[k] = (1.0 - lam) * va + lam * vb
            else:
                merged[k] = va
        shell.load_state_dict(merged, strict=True)
        del merged
        vloss = evaluate_lm_loss(shell, val_ld, device, max_batches=max_b)
        losses[lam] = vloss
        print(f"  [{method_name}] lambda={lam:.1f}  val_loss={vloss:.4f}")

    del shell
    if device.type == "cuda":
        torch.cuda.empty_cache()

    l0, l1 = losses[0.0], losses[1.0]
    barrier = max(lo - ((1.0 - lam) * l0 + lam * l1) for lam, lo in losses.items())
    return {
        "method": method_name,
        "loss_0": l0,
        "loss_50": losses[0.5],
        "loss_1": l1,
        "barrier": float(barrier),
        "loss_curve": losses,
    }

def print_summary(rows: List[Dict]):
    print("\n" + "=" * 92)
    print(f"{'Method':<36} {'Loss@0':>9} {'Loss@0.5':>9} {'Loss@1':>9} {'Barrier':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['method']:<36} {r['loss_0']:9.4f} {r['loss_50']:9.4f} "
              f"{r['loss_1']:9.4f} {r['barrier']:10.4f}")
    print("=" * 92)
    print("Barrier = max_lambda[Loss(lambda) - ((1-lambda)Loss0 + lambda Loss1)]  (lower is better)")

def plot_results(rows: List[Dict], out_dir: Path):
    """Test-loss-vs-lambda curves + barrier bar chart."""
    style = {
        "Vanilla averaging": dict(color="#2ca02c", marker="D"),
        "Activation matching": dict(color="#9467bd", marker="s"),
        "Weight matching (ours)": dict(color="#ff7f0e", marker="v"),
        "Learned matching (permutations)": dict(color="#d62728", marker="^"),
        "Learned matching (ours)": dict(color="#1f77b4", marker="o"),
    }


    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for r in rows:
        st = style.get(r["method"], dict(color="gray", marker="o"))
        xs = sorted(r["loss_curve"])
        ys = [r["loss_curve"][x] for x in xs]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8,
                label=r["method"])
    ax.set_xlabel(r"Interpolation coefficient ($\lambda$)")
    ax.set_ylabel("Test loss")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.subplots_adjust(right=0.62)
    png = out_dir / "loss_vs_lambda.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "loss_vs_lambda.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {png}")


    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    names = [r["method"] for r in rows]
    vals = [r["barrier"] for r in rows]
    colors = [style.get(n, {}).get("color", "gray") for n in names]
    ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Test-loss barrier")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    png2 = out_dir / "barrier_bars.png"
    fig.savefig(png2, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "barrier_bars.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {png2}")


    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, [r["loss_0"] for r in rows], w, label=r"$\lambda=0$ (Deep)", color="#4c78a8")
    ax.bar(x, [r["loss_50"] for r in rows], w, label=r"$\lambda=0.5$", color="#f58518")
    ax.bar(x + w, [r["loss_1"] for r in rows], w, label=r"$\lambda=1$ (Expanded)", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Test loss")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    png3 = out_dir / "loss_endpoints_mid.png"
    fig.savefig(png3, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "loss_endpoints_mid.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {png3}")


def reclaim(*objs, tag=""):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        free, total = torch.cuda.mem_get_info()
        print(f"[reclaim:{tag}] GPU free={free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    elif tag:
        print(f"[reclaim:{tag}] done", flush=True)

def park_model(model, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        "depth": int(model.depth),
        "is_identity": [bool(b.is_identity) for b in model.blocks],
    }
    torch.save(payload, path)
    return path

def load_parked(path: Path, device):
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    m = build_gpt(ckpt["depth"], CFG)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    for blk, flag in zip(m.blocks, ckpt.get("is_identity", [])):
        blk.is_identity = bool(flag)
    return m.to(device).eval()


train_loader, stat_loader, val_loader = get_token_loaders(
    DEVICE, batch_size=BATCH_SIZE, eval_batch=EVAL_BATCH, seq_len=SEQ_LEN,
)


if SKIP_TRAIN or (CKPT_DEEP and CKPT_SHALLOW):
    path_deep = Path(CKPT_DEEP)
    path_shal = Path(CKPT_SHALLOW)
    print(f"[reuse] {path_deep}\n[reuse] {path_shal}", flush=True)
else:
    _deep_m = build_gpt(CFG.deep_depth, CFG)
    path_deep = train_one(_deep_m, train_loader, val_loader, DEVICE, SEED_DEEP, "deep12", CFG)
    reclaim(_deep_m, tag="after_deep_train")

    _shal_m = build_gpt(CFG.shallow_depth, CFG)
    path_shal = train_one(_shal_m, train_loader, val_loader, DEVICE, SEED_SHALLOW, "shallow6", CFG)
    reclaim(_shal_m, tag="after_shallow_train")

def _ckpt_file(p: Path) -> Path:
    p = Path(p)
    c = p / "checkpoint.pt"
    return c if c.is_file() else p

_deep = build_gpt(CFG.deep_depth, CFG)
_deep.load_state_dict(
    torch.load(_ckpt_file(path_deep), map_location="cpu", weights_only=False)["state_dict"]
)
park_deep = park_model(_deep, PARK / "deep12.pt")
reclaim(_deep, tag="park_deep")

_shallow = build_gpt(CFG.shallow_depth, CFG)
_shallow.load_state_dict(
    torch.load(_ckpt_file(path_shal), map_location="cpu", weights_only=False)["state_dict"]
)
_expanded = expand_shallow_gpt2(_shallow.cpu(), CFG.deep_depth)
assert _expanded.depth == 12
for _i, _blk in enumerate(_expanded.blocks):
    assert _blk.is_identity == (_i % 2 == 1)
park_exp = park_model(_expanded, PARK / "expanded12.pt")
reclaim(_shallow, _expanded, tag="park_expanded")
print(f"[park] {park_deep}\n[park] {park_exp}", flush=True)


_m = load_parked(park_deep, DEVICE)
_l_d = evaluate_lm_loss(_m, val_loader, DEVICE, max_batches=(2 if MOCK_TRAINING else None))
print(f"[endpoint] Deep12     val_loss={_l_d:.4f}  ppl={math.exp(min(_l_d, 20.0)):.1f}", flush=True)
reclaim(_m, tag="ep_deep")

_m = load_parked(park_exp, DEVICE)
_l_e = evaluate_lm_loss(_m, val_loader, DEVICE, max_batches=(2 if MOCK_TRAINING else None))
print(f"[endpoint] Expanded12 val_loss={_l_e:.4f}  ppl={math.exp(min(_l_e, 20.0)):.1f}", flush=True)
reclaim(_m, tag="ep_exp")


results = []

def _barrier(name, path_b):
    _A = load_parked(park_deep, DEVICE)
    _B = load_parked(path_b, DEVICE)
    row = measure_barrier(_A, _B, val_loader, DEVICE, name)
    reclaim(_A, _B, tag=f"bar:{name[:18]}")
    return row

print("\n### 1 Vanilla averaging", flush=True)
results.append(_barrier("Vanilla averaging", park_exp))

print("\n### 2 Activation matching", flush=True)
_A = load_parked(park_deep, DEVICE)
_B = load_parked(park_exp, DEVICE)
_B_act = activation_match(_A, _B, stat_loader, DEVICE, CFG, max_tokens=ACT_MAX_TOKENS)
park_act = park_model(_B_act, PARK / "aligned_act.pt")
reclaim(_A, _B, _B_act, tag="after_act")
results.append(_barrier("Activation matching", park_act))

print("\n### 3 Weight matching (ours)", flush=True)
_A = load_parked(park_deep, DEVICE)
_B = load_parked(park_exp, DEVICE)
_B_wm = weight_match(_A, _B, CFG)
park_wm = park_model(_B_wm, PARK / "aligned_wm.pt")
reclaim(_A, _B, _B_wm, tag="after_wm")
results.append(_barrier("Weight matching (ours)", park_wm))

print("\n### 4 Learned matching (permutations)", flush=True)
_A = load_parked(park_deep, DEVICE)
_B = load_parked(park_wm, DEVICE)
_B_perm = train_learned_matching(_A, _B, train_loader, DEVICE, True, CFG)
park_perm = park_model(_B_perm, PARK / "aligned_perm.pt")
reclaim(_A, _B, _B_perm, tag="after_perm")
results.append(_barrier("Learned matching (permutations)", park_perm))

print("\n### 5 Learned matching (ours)", flush=True)
_A = load_parked(park_deep, DEVICE)
_B = load_parked(park_wm, DEVICE)
_B_full = train_learned_matching(_A, _B, train_loader, DEVICE, False, CFG)
park_full = park_model(_B_full, PARK / "aligned_full.pt")
reclaim(_A, _B, _B_full, tag="after_full")
results.append(_barrier("Learned matching (ours)", park_full))

print_summary(results)


plot_results(results, PLOTS)
out_json = PLOTS.parent / "barrier_results.json"
with open(out_json, "w", encoding="utf-8") as _f:
    json.dump(
        {
            "mock": MOCK_TRAINING,
            "results": [
                {
                    "method": r["method"],
                    "loss_0": r["loss_0"],
                    "loss_50": r["loss_50"],
                    "loss_1": r["loss_1"],
                    "barrier": r["barrier"],
                    "loss_curve": {str(k): float(v) for k, v in r["loss_curve"].items()},
                }
                for r in results
            ],
        },
        _f,
        indent=2,
    )


assert out_json.exists()


import shutil
assert out_json.exists()
_zip_base = OUTPUT_ROOT.parent / "shakespeare_depth_het_bundle"
_zip_path = shutil.make_archive(str(_zip_base), "zip", root_dir=str(OUTPUT_ROOT))
_size_mb = os.path.getsize(_zip_path) / 1e6
with open(_zip_path, "rb") as _fz:
    _blob = _fz.read()
