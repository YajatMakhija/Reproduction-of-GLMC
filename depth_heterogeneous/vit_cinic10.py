"""Depth-heterogeneous ViT merging on CINIC-10 (GLMC reproduction + extension)."""

import copy
import gc
import json
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
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


SMOKE = True
USE_BF16 = True
SEED_DEEP, SEED_SHALLOW = 0, 1
CKPT_DEEP = None
CKPT_SHALLOW = None
SKIP_TRAIN = False

BATCH_SIZE = 128
EVAL_BATCH = 2048
EPOCHS = 150
WM_ITERS = 15
ACT_BATCHES = 24
ACT_MAX_TOKENS = 4096
LEARNED_EPOCHS = 2
LEARNED_BATCHES = 200

OUTPUT_ROOT = Path("outputs/cinic10_depth_het_marimo")
PARK = OUTPUT_ROOT / "park"
PLOTS = OUTPUT_ROOT / "plots"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if SMOKE:
    EPOCHS = 1
    EVAL_BATCH = 128
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
    torch.backends.cudnn.benchmark = True
    _free, _total = torch.cuda.mem_get_info()
    print(
        f"[gpu] {torch.cuda.get_device_name(0)}  "
        f"free={_free/1e9:.1f}/{_total/1e9:.1f} GB  torch={torch.__version__}",
        flush=True,
    )
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PARK.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


CINIC_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC_STD = (0.24205776, 0.23828046, 0.25874835)
NUM_CLASSES = 10

@dataclass
class TrainCfg:
    patch_size: int = 4
    dim: int = 256
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    bias: bool = True
    deep_depth: int = 12
    shallow_depth: int = 6
    batch_size: int = 128
    epochs: int = 150
    lr: float = 3e-4
    weight_decay: float = 1e-3
    use_bf16: bool = True
    wm_iterations: int = 15
    act_batches: int = 20
    learned_epochs: int = 2
    learned_batches_per_epoch: int = 200
    narrow_lambda_lo: float = 0.4
    narrow_lambda_hi: float = 0.6

CFG = TrainCfg(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    use_bf16=bool(USE_BF16),
    wm_iterations=WM_ITERS,
    act_batches=ACT_BATCHES,
    learned_epochs=LEARNED_EPOCHS,
    learned_batches_per_epoch=LEARNED_BATCHES,
)
print("[cfg]", CFG, " MOCK=", MOCK_TRAINING, flush=True)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm: RMSNorm(Ox) = O * RMSNorm(x) for orthogonal O."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.dim, self.dim_head = heads, dim, dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        attn = self.drop((q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1))
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.drop(self.to_out(out))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, bias):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout, bias)
        self.norm2 = RMSNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout, bias)
        self.is_identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return x
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    @torch.no_grad()
    def make_identity(self):
        """Silent block: bypass Attn/MLP residuals and zero their output maps."""
        self.is_identity = True
        nn.init.zeros_(self.attn.to_qkv.weight)
        nn.init.zeros_(self.attn.to_out.weight)
        if self.attn.to_out.bias is not None:
            nn.init.zeros_(self.attn.to_out.bias)
        nn.init.zeros_(self.mlp.fc1.weight)
        nn.init.zeros_(self.mlp.fc2.weight)
        if self.mlp.fc1.bias is not None:
            nn.init.zeros_(self.mlp.fc1.bias)
        if self.mlp.fc2.bias is not None:
            nn.init.zeros_(self.mlp.fc2.bias)

class RMSViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=12,
        heads=8,
        mlp_dim=512,
        channels=3,
        dropout=0.1,
        bias=True,
    ):
        super().__init__()
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        n_patches = (ih // ph) * (iw // pw)
        patch_dim = channels * ph * pw
        dim_head = dim // heads

        self.dim, self.depth, self.heads, self.mlp_dim = dim, depth, heads, mlp_dim

        self.to_patch = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw),
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(n_patches, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout, bias)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes, bias=bias)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b = img.size(0)
        x = self.to_patch(img)
        n = x.size(1)
        pos = torch.arange(n, device=img.device).unsqueeze(0).expand(b, -1)
        x = x + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.final_norm(x).mean(dim=1))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

def build_vit(depth: int, cfg: TrainCfg = CFG) -> RMSViT:
    return RMSViT(
        image_size=32, patch_size=cfg.patch_size, num_classes=NUM_CLASSES,
        dim=cfg.dim, depth=depth, heads=cfg.heads, mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout, bias=cfg.bias,
    )


def _kaggle_root() -> Path:
    env = os.environ.get("CINIC10_ROOT")
    if env and (Path(env) / "train").is_dir():
        return Path(env)
    import kagglehub
    p = Path(kagglehub.dataset_download("mengcius/cinic10"))

    def looks(q: Path) -> bool:
        return (q / "train").is_dir() and (q / "test").is_dir()

    if looks(p):
        return p
    for c in p.rglob("*"):
        if c.is_dir() and looks(c):
            return c
    raise FileNotFoundError(f"CINIC-10 not under {p}")

def _decode_split(split_dir: Path, cache_path: Path):
    """One-time PNG decode into a uint8 tensor, cached on disk.
    transform must be module-level (pil_to_tensor) so spawned DataLoader
    workers can pickle the dataset."""
    import torchvision.transforms.functional as TF
    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        return blob["images"], blob["labels"]
    ds = torchvision.datasets.ImageFolder(str(split_dir), transform=TF.pil_to_tensor)
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=3)
    imgs, lbls = [], []
    for xb, yb in loader:
        imgs.append(xb)
        lbls.append(yb)
    images, labels = torch.cat(imgs), torch.cat(lbls)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, cache_path)
    print(f"[data] decoded {Path(split_dir).name}: {tuple(images.shape)} -> {cache_path}")
    return images, labels

class GPULoader:
    """DataLoader stand-in over GPU-resident uint8 images. Yields (x, y)
    already on device; augmentation (kornia, per-sample) runs on GPU."""

    def __init__(self, images, labels, batch_size, shuffle, augment=None, drop_last=False):
        self.images, self.labels = images, labels
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.augment = augment
        dev = images.device
        self.mean = torch.tensor(CINIC_MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(CINIC_STD, device=dev).view(1, 3, 1, 1)

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

def get_gpu_loaders(device, batch_size=128, eval_batch=2048, cache_dir="cinic10_cache"):
    """Returns (train_loader [augmented], stat_loader [no aug], test_loader)."""
    import kornia.augmentation as K
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    try:
        root = _kaggle_root()
        tr_i, tr_l = _decode_split(root / "train", cache / "train.pt")
        te_i, te_l = _decode_split(root / "test", cache / "test.pt")
    except Exception as e:
        if not MOCK_TRAINING:
            raise
        print(f"[data] kagglehub unavailable ({e}); synthetic MOCK tensors")
        g = torch.Generator().manual_seed(0)
        tr_i = torch.randint(0, 256, (2048, 3, 32, 32), dtype=torch.uint8, generator=g)
        te_i = torch.randint(0, 256, (1024, 3, 32, 32), dtype=torch.uint8, generator=g)
        tr_l = torch.randint(0, 10, (2048,), generator=g)
        te_l = torch.randint(0, 10, (1024,), generator=g)

    tr_i, tr_l = tr_i.to(device), tr_l.to(device)
    te_i, te_l = te_i.to(device), te_l.to(device)
    augment = nn.Sequential(
        K.RandomCrop((32, 32), padding=4),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJiggle(0.4, 0.4, 0.4, 0.1, p=1.0),
    ).to(device)
    train_ld = GPULoader(tr_i, tr_l, batch_size, shuffle=True, augment=augment, drop_last=True)
    stat_ld = GPULoader(tr_i, tr_l, batch_size, shuffle=True)
    test_ld = GPULoader(te_i, te_l, eval_batch, shuffle=False)
    print(f"[data] CINIC-10 GPU-resident on {device}: "
          f"train={tr_i.shape[0]:,}  test={te_i.shape[0]:,}")
    return train_ld, stat_ld, test_ld


@torch.no_grad()
def evaluate_acc_loss(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda" and CFG.use_bf16
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if max_batches is not None and i + 1 >= max_batches:
            break
        if MOCK_TRAINING:
            break
    return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)

def train_one(model, train_ld, test_ld, device, seed, tag, cfg=CFG) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    crit = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda" and cfg.use_bf16


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
            for _ in range(start_ep):
                sched.step()
            print(f"[train] {tag}: resumed at epoch {start_ep}/{cfg.epochs}")

    acc = tloss = 0.0
    if history:
        acc = history[-1].get("test_acc", 0.0)
        tloss = history[-1].get("test_loss", 0.0)
    for ep in range(start_ep, cfg.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for x, y in tqdm(train_ld, desc=f"{tag} ep{ep+1}/{cfg.epochs}", leave=False):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            run_n += x.size(0)
            if MOCK_TRAINING:
                break
        sched.step()
        max_b = 2 if MOCK_TRAINING else None
        acc, tloss = evaluate_acc_loss(model, test_ld, device, max_batches=max_b)
        history.append({"epoch": ep + 1, "train_loss": run_loss / max(run_n, 1),
                        "test_acc": acc, "test_loss": tloss})
        log_path.write_text(json.dumps(history, indent=2))
        torch.save({
            "state_dict": model.state_dict(), "depth": model.depth,
            "cfg": asdict(cfg), "tag": tag, "seed": seed,
            "test_acc": acc, "test_loss": tloss,
        }, ckpt_path)
        torch.save({"opt": opt.state_dict(), "epoch": ep + 1}, resume_path)
        print(f"[train] {tag} ep{ep+1}: test_acc={acc:.2f}%  test_loss={tloss:.4f}")
    return out


@torch.no_grad()
def expand_shallow_vit(shallow: RMSViT, target_depth: int = 12) -> RMSViT:
    """Map 6 trained blocks -> even indices {0,2,...,10}; odd slots = identity."""
    assert shallow.depth * 2 == target_depth
    out = build_vit(target_depth, CFG)
    out.to_patch.load_state_dict(shallow.to_patch.state_dict())
    out.pos_emb.load_state_dict(shallow.pos_emb.state_dict())
    out.final_norm.load_state_dict(shallow.final_norm.state_dict())
    out.head.load_state_dict(shallow.head.state_dict())

    for s_i, blk in enumerate(shallow.blocks):
        d_i = 2 * s_i
        out.blocks[d_i].load_state_dict(blk.state_dict())
        out.blocks[d_i].is_identity = False

    for d_i in range(1, target_depth, 2):
        out.blocks[d_i].make_identity()


    x = torch.randn(2, 64, CFG.dim)
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
def apply_residual_O(model: RMSViT, O: torch.Tensor):
    """Writers: W ← O @ W; readers: W ← W @ O^T."""
    lin = model.to_patch[2]
    lin.weight.data = O @ lin.weight.data
    if lin.bias is not None:
        lin.bias.data = O @ lin.bias.data
    model.pos_emb.weight.data = model.pos_emb.weight.data @ O.T

    for blk in model.blocks:
        if blk.is_identity:
            continue
        blk.attn.to_qkv.weight.data = blk.attn.to_qkv.weight.data @ O.T
        blk.attn.to_out.weight.data = O @ blk.attn.to_out.weight.data
        if blk.attn.to_out.bias is not None:
            blk.attn.to_out.bias.data = O @ blk.attn.to_out.bias.data
        blk.mlp.fc1.weight.data = blk.mlp.fc1.weight.data @ O.T
        blk.mlp.fc2.weight.data = O @ blk.mlp.fc2.weight.data
        if blk.mlp.fc2.bias is not None:
            blk.mlp.fc2.bias.data = O @ blk.mlp.fc2.bias.data

    model.head.weight.data = model.head.weight.data @ O.T

@torch.no_grad()
def permute_mlp_channels(blk: TransformerBlock, P: torch.Tensor):
    if blk.is_identity:
        return
    blk.mlp.fc1.weight.data = P @ blk.mlp.fc1.weight.data
    if blk.mlp.fc1.bias is not None:
        blk.mlp.fc1.bias.data = P @ blk.mlp.fc1.bias.data
    blk.mlp.fc2.weight.data = blk.mlp.fc2.weight.data @ P.T

def _perm_heads_qkv(W, P, H, Dh, D):

    t = W.view(3, H, Dh, D)
    t = torch.einsum("hk,ckdD->chdD", P, t)
    return t.reshape(3 * H * Dh, D)

def _perm_heads_out(W, P, H, Dh, D):

    t = W.view(D, H, Dh).permute(1, 0, 2).reshape(H, D * Dh)
    t = P @ t
    return t.view(H, D, Dh).permute(1, 0, 2).contiguous().view(D, H * Dh)

@torch.no_grad()
def permute_attention_heads(blk: TransformerBlock, P_heads: torch.Tensor):
    if blk.is_identity:
        return
    H, Dh, D = blk.attn.heads, blk.attn.dim_head, blk.attn.dim
    P = P_heads.to(dtype=blk.attn.to_qkv.weight.dtype)
    blk.attn.to_qkv.weight.data = _perm_heads_qkv(blk.attn.to_qkv.weight.data, P, H, Dh, D)
    blk.attn.to_out.weight.data = _perm_heads_out(blk.attn.to_out.weight.data, P, H, Dh, D)


@torch.no_grad()
def _collect_acts(model: RMSViT, loader, device, max_batches: int, max_tokens: int = 8192):
    """Collect capped activation banks; keep tensors on GPU then one .cpu() cat.

    Per-layer token cap avoids host RAM spikes from every patch × batch.
    """
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

    for bi, (imgs, _) in enumerate(loader):
        if bi >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=False)
        x = model.to_patch(imgs)
        bsz, npatch, _ = x.shape
        pos = torch.arange(npatch, device=device).unsqueeze(0).expand(bsz, -1)
        x = x + model.pos_emb(pos)
        for i, blk in enumerate(model.blocks):
            if blk.is_identity:
                x = blk(x)
                continue
            h = blk.norm1(x)
            q, k, v = blk.attn.to_qkv(h).chunk(3, dim=-1)
            q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=blk.attn.heads) for t in (q, k, v))
            hout = ((q @ k.transpose(-1, -2) * blk.attn.scale).softmax(-1) @ v)
            head_acts[i].append(hout.abs().mean((2, 3)).reshape(-1, blk.attn.heads).float())
            x = blk(x)
        residual.append(x.mean(1).float())
        del imgs, x
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
def activation_match(model_a, model_b, loader, device, cfg, max_tokens: int = 8192) -> RMSViT:
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
def _paired_residual_stacks(model_a: RMSViT, model_b: RMSViT):
    """Build equal-width residual stacks using only layers active in both models."""
    cols_a = [model_a.to_patch[2].weight.data, model_a.pos_emb.weight.data.T]
    cols_b = [model_b.to_patch[2].weight.data, model_b.pos_emb.weight.data.T]
    for ba, bb in zip(model_a.blocks, model_b.blocks):
        if ba.is_identity or bb.is_identity:
            continue
        cols_a += [
            ba.attn.to_out.weight.data, ba.mlp.fc2.weight.data,
            ba.attn.to_qkv.weight.data.T, ba.mlp.fc1.weight.data.T,
        ]
        cols_b += [
            bb.attn.to_out.weight.data, bb.mlp.fc2.weight.data,
            bb.attn.to_qkv.weight.data.T, bb.mlp.fc1.weight.data.T,
        ]
    cols_a.append(model_a.head.weight.data.T)
    cols_b.append(model_b.head.weight.data.T)
    return torch.cat(cols_a, 1), torch.cat(cols_b, 1)

@torch.no_grad()
def weight_match(model_a: RMSViT, model_b: RMSViT, cfg: TrainCfg) -> RMSViT:
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
                Wqkv = blk.attn.to_qkv.weight.data.view(3, H, Dh, D)
                Wout = blk.attn.to_out.weight.data.view(D, H, Dh).permute(1, 2, 0)
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
def hard_align(model_b: RMSViT, O, P_heads, P_mlp) -> RMSViT:
    B = copy.deepcopy(model_b)
    apply_residual_O(B, O.detach())
    for i, blk in enumerate(B.blocks):
        if blk.is_identity:
            continue
        permute_mlp_channels(blk, P_mlp[i].detach())
        permute_attention_heads(blk, P_heads[i].detach())
    return B

def ste_interpolated_logits(
    model_a: RMSViT,
    model_b: RMSViT,
    align: AlignmentState,
    x: torch.Tensor,
    lam: float,
    soft: bool = False,
) -> torch.Tensor:
    """Blend weights with (soft/STE) P and SVD-projected O; keep graph into align latents."""
    O, P_heads, P_mlp = align.projected(soft=soft)
    if align.permutations_only:
        O = torch.eye(CFG.dim, device=x.device, dtype=model_a.to_patch[2].weight.dtype)

    H, Dh, D = CFG.heads, CFG.dim // CFG.heads, CFG.dim


    tokens = model_a.to_patch[0](x)
    tokens = model_a.to_patch[1](tokens)
    Wa, Wb = model_a.to_patch[2].weight, model_b.to_patch[2].weight
    ba = model_a.to_patch[2].bias
    bb = model_b.to_patch[2].bias
    W_patch = (1.0 - lam) * Wa + lam * (O @ Wb)
    b_patch = None
    if ba is not None and bb is not None:
        b_patch = (1.0 - lam) * ba + lam * (O @ bb)
    tokens = F.linear(tokens, W_patch, b_patch)
    tokens = model_a.to_patch[3](tokens)

    bsz, npatch, _ = tokens.shape
    pos = torch.arange(npatch, device=x.device).unsqueeze(0).expand(bsz, -1)

    E = (1.0 - lam) * model_a.pos_emb.weight + lam * (model_b.pos_emb.weight @ O.T)
    h = tokens + F.embedding(pos, E)


    for i in range(model_a.depth):
        ba_blk, bb_blk = model_a.blocks[i], model_b.blocks[i]


        Wqkv_a = ba_blk.attn.to_qkv.weight
        Wqkv_b = _perm_heads_qkv(bb_blk.attn.to_qkv.weight @ O.T, P_heads[i], H, Dh, D)
        Wqkv = (1.0 - lam) * Wqkv_a + lam * Wqkv_b

        Wout_a = ba_blk.attn.to_out.weight
        Wout_b = _perm_heads_out(O @ bb_blk.attn.to_out.weight, P_heads[i], H, Dh, D)
        Wout = (1.0 - lam) * Wout_a + lam * Wout_b
        bout = None
        if ba_blk.attn.to_out.bias is not None and bb_blk.attn.to_out.bias is not None:
            bout = (1.0 - lam) * ba_blk.attn.to_out.bias + lam * (O @ bb_blk.attn.to_out.bias)


        hn = ba_blk.norm1(h)
        qkv = F.linear(hn, Wqkv, None).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (hh d) -> b hh n d", hh=H) for t in qkv)
        attn = (q @ k.transpose(-1, -2) * ba_blk.attn.scale).softmax(dim=-1)
        attn = ba_blk.attn.drop(attn)
        aout = rearrange(attn @ v, "b hh n d -> b n (hh d)")
        h = h + ba_blk.attn.drop(F.linear(aout, Wout, bout))


        W1_a, W1_b = ba_blk.mlp.fc1.weight, bb_blk.mlp.fc1.weight
        W1 = (1.0 - lam) * W1_a + lam * (P_mlp[i] @ (W1_b @ O.T))
        b1 = None
        if ba_blk.mlp.fc1.bias is not None and bb_blk.mlp.fc1.bias is not None:
            b1 = (1.0 - lam) * ba_blk.mlp.fc1.bias + lam * (P_mlp[i] @ bb_blk.mlp.fc1.bias)

        W2_a, W2_b = ba_blk.mlp.fc2.weight, bb_blk.mlp.fc2.weight
        W2 = (1.0 - lam) * W2_a + lam * (O @ (W2_b @ P_mlp[i].T))
        b2 = None
        if ba_blk.mlp.fc2.bias is not None and bb_blk.mlp.fc2.bias is not None:
            b2 = (1.0 - lam) * ba_blk.mlp.fc2.bias + lam * (O @ bb_blk.mlp.fc2.bias)

        hn = ba_blk.norm2(h)
        h = h + ba_blk.mlp.drop(F.linear(ba_blk.mlp.drop(ba_blk.mlp.act(F.linear(hn, W1, b1))), W2, b2))

    h = model_a.final_norm(h).mean(dim=1)
    Wh = (1.0 - lam) * model_a.head.weight + lam * (model_b.head.weight @ O.T)
    bh = None
    if model_a.head.bias is not None and model_b.head.bias is not None:
        bh = (1.0 - lam) * model_a.head.bias + lam * model_b.head.bias
    return F.linear(h, Wh, bh)

def train_learned_matching(
    model_a, model_b_wm, train_ld, device, permutations_only: bool, cfg: TrainCfg,
) -> RMSViT:
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
            x, y = x.to(device), y.to(device)
            lam = random.uniform(cfg.narrow_lambda_lo, cfg.narrow_lambda_hi)
            opt.zero_grad(set_to_none=True)
            logits = ste_interpolated_logits(model_a, model_b_wm, align, x, lam,
                                             soft=permutations_only)
            loss = F.cross_entropy(logits, y)
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
        return hard_align(model_b_wm, O, Ph, Pm)


@torch.no_grad()
def measure_barrier(model_a, aligned_b, test_ld, device, method_name: str) -> Dict:
    """
    Barrier = max_lambda [ ((1-lambda) Acc(0) + lambda Acc(1)) - Acc(lambda) ]
    with W(lambda) = (1-lambda) W_A + lambda W_B_aligned.
    """
    model_a, aligned_b = model_a.to(device).eval(), aligned_b.to(device).eval()

    sd_a = model_a.state_dict()
    sd_b = aligned_b.state_dict()
    shell = copy.deepcopy(model_a).to(device)
    coeffs = [round(0.1 * i, 1) for i in range(11)]
    accs = {}
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
        acc, tloss = evaluate_acc_loss(shell, test_ld, device, max_batches=max_b)
        accs[lam] = acc
        losses[lam] = tloss
        print(f"  [{method_name}] lambda={lam:.1f}  acc={acc:.2f}%  loss={tloss:.4f}")

    del shell
    if device.type == "cuda":
        torch.cuda.empty_cache()

    a0, a1 = accs[0.0], accs[1.0]
    barrier = max(((1.0 - lam) * a0 + lam * a1) - acc for lam, acc in accs.items())

    l0, l1 = losses[0.0], losses[1.0]
    loss_barrier = max(lo - ((1.0 - lam) * l0 + lam * l1) for lam, lo in losses.items())
    return {
        "method": method_name,
        "acc_0": a0,
        "acc_50": accs[0.5],
        "acc_1": a1,
        "barrier": float(barrier),
        "curve": accs,
        "loss_0": l0,
        "loss_50": losses[0.5],
        "loss_1": l1,
        "loss_barrier": float(loss_barrier),
        "loss_curve": losses,
    }

def print_summary(rows: List[Dict]):
    print("\n" + "=" * 110)
    print(f"{'Method':<36} {'Acc@0':>8} {'Acc@0.5':>8} {'Acc@1':>8} {'AccBar':>9} "
          f"{'Loss@0.5':>9} {'LossBar':>9}")
    print("-" * 110)
    for r in rows:
        print(f"{r['method']:<36} {r['acc_0']:8.2f} {r['acc_50']:8.2f} "
              f"{r['acc_1']:8.2f} {r['barrier']:9.4f} "
              f"{r['loss_50']:9.4f} {r['loss_barrier']:9.4f}")
    print("=" * 110)
    print("AccBar  = max_lambda[((1-lambda)Acc0 + lambda Acc1) - Acc(lambda)]  (lower is better)")
    print("LossBar = max_lambda[Loss(lambda) - ((1-lambda)Loss0 + lambda Loss1)]  (lower is better)")

def plot_results(rows: List[Dict], out_dir: Path):
    """Acc-vs-lambda curves + barrier bar chart + endpoint/midpoint bars."""
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
        xs = sorted(r["curve"])
        ys = [r["curve"][x] for x in xs]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8,
                label=r["method"])
    ax.set_xlabel(r"Interpolation coefficient ($\lambda$)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.subplots_adjust(right=0.62)
    png = out_dir / "acc_vs_lambda.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "acc_vs_lambda.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {png}")


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
    png_l = out_dir / "loss_vs_lambda.png"
    fig.savefig(png_l, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "loss_vs_lambda.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] {png_l}")


    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    names = [r["method"] for r in rows]
    vals = [r["barrier"] for r in rows]
    colors = [style.get(n, {}).get("color", "gray") for n in names]
    ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy barrier (pp)")
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
    ax.bar(x - w, [r["acc_0"] for r in rows], w, label=r"$\lambda=0$ (Deep)", color="#4c78a8")
    ax.bar(x, [r["acc_50"] for r in rows], w, label=r"$\lambda=0.5$", color="#f58518")
    ax.bar(x + w, [r["acc_1"] for r in rows], w, label=r"$\lambda=1$ (Expanded)", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    png3 = out_dir / "acc_endpoints_mid.png"
    fig.savefig(png3, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "acc_endpoints_mid.pdf", bbox_inches="tight", facecolor="white")
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
    m = build_vit(ckpt["depth"], CFG)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    for blk, flag in zip(m.blocks, ckpt.get("is_identity", [])):
        blk.is_identity = bool(flag)
    return m.to(device).eval()


train_loader, stat_loader, test_loader = get_gpu_loaders(
    DEVICE, batch_size=BATCH_SIZE, eval_batch=EVAL_BATCH,
)


if SKIP_TRAIN or (CKPT_DEEP and CKPT_SHALLOW):
    path_deep = Path(CKPT_DEEP)
    path_shal = Path(CKPT_SHALLOW)
    print(f"[reuse] {path_deep}\n[reuse] {path_shal}", flush=True)
else:
    _deep_m = build_vit(CFG.deep_depth, CFG)
    path_deep = train_one(_deep_m, train_loader, test_loader, DEVICE, SEED_DEEP, "deep12", CFG)
    reclaim(_deep_m, tag="after_deep_train")

    _shal_m = build_vit(CFG.shallow_depth, CFG)
    path_shal = train_one(_shal_m, train_loader, test_loader, DEVICE, SEED_SHALLOW, "shallow6", CFG)
    reclaim(_shal_m, tag="after_shallow_train")

def _ckpt_file(p: Path) -> Path:
    p = Path(p)
    c = p / "checkpoint.pt"
    return c if c.is_file() else p

_deep = build_vit(CFG.deep_depth, CFG)
_deep.load_state_dict(
    torch.load(_ckpt_file(path_deep), map_location="cpu", weights_only=False)["state_dict"]
)
park_deep = park_model(_deep, PARK / "deep12.pt")
reclaim(_deep, tag="park_deep")

_shallow = build_vit(CFG.shallow_depth, CFG)
_shallow.load_state_dict(
    torch.load(_ckpt_file(path_shal), map_location="cpu", weights_only=False)["state_dict"]
)
_expanded = expand_shallow_vit(_shallow.cpu(), CFG.deep_depth)
assert _expanded.depth == 12
for _i, _blk in enumerate(_expanded.blocks):
    assert _blk.is_identity == (_i % 2 == 1)
park_exp = park_model(_expanded, PARK / "expanded12.pt")
reclaim(_shallow, _expanded, tag="park_expanded")
print(f"[park] {park_deep}\n[park] {park_exp}", flush=True)


_m = load_parked(park_deep, DEVICE)
_acc_d, _loss_d = evaluate_acc_loss(
    _m, test_loader, DEVICE, max_batches=(2 if MOCK_TRAINING else None),
)
print(f"[endpoint] Deep12     acc={_acc_d:.2f}%  loss={_loss_d:.4f}", flush=True)
reclaim(_m, tag="ep_deep")

_m = load_parked(park_exp, DEVICE)
_acc_e, _loss_e = evaluate_acc_loss(
    _m, test_loader, DEVICE, max_batches=(2 if MOCK_TRAINING else None),
)
print(f"[endpoint] Expanded12 acc={_acc_e:.2f}%  loss={_loss_e:.4f}", flush=True)
reclaim(_m, tag="ep_exp")


results = []

def _barrier(name, path_b):
    _A = load_parked(park_deep, DEVICE)
    _B = load_parked(path_b, DEVICE)
    row = measure_barrier(_A, _B, test_loader, DEVICE, name)
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
                    "acc_0": r["acc_0"],
                    "acc_50": r["acc_50"],
                    "acc_1": r["acc_1"],
                    "barrier": r["barrier"],
                    "curve": {str(k): float(v) for k, v in r["curve"].items()},
                    "loss_0": r["loss_0"],
                    "loss_50": r["loss_50"],
                    "loss_1": r["loss_1"],
                    "loss_barrier": r["loss_barrier"],
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
_zip_base = OUTPUT_ROOT.parent / "cinic10_depth_het_bundle"
_zip_path = shutil.make_archive(str(_zip_base), "zip", root_dir=str(OUTPUT_ROOT))
_size_mb = os.path.getsize(_zip_path) / 1e6
with open(_zip_path, "rb") as _fz:
    _blob = _fz.read()
