"""Function-preservation checks for the whole pipeline (no training needed):
reparameterization, weight matching, and the merger must all preserve outputs."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_vit, reparameterize
from merger import ViTMerger
from weight_matching import weight_matching

CFG = dict(patch_size=4, dim=64, depth=3, heads=4, mlp_dim=128, dropout=0.0, bias=True)


def rand(seed):
    torch.manual_seed(seed)
    m = build_vit(CFG, 10).eval()
    for n, p in m.named_parameters():          # make norms/biases non-trivial
        if "bias" in n:
            torch.nn.init.normal_(p, 0.0, 0.2)
        elif "norm" in n.lower() and "weight" in n:
            torch.nn.init.normal_(p, 1.0, 0.2)
    return m


x = torch.randn(4, 3, 32, 32)
lnA, lnB = rand(1), rand(2)
A, B = reparameterize(lnA, 10).eval(), reparameterize(lnB, 10).eval()

d_rep = (lnA(x)[0] - A(x)[0]).abs().max().item()
print(f"[reparam]  max|Δ| = {d_rep:.2e}")
assert d_rep < 1e-3

outA, outB = A(x)[0].clone(), B(x)[0].clone()
weight_matching(A, B, CFG["heads"], iterations=8)
d_wm = (B(x)[0] - outB).abs().max().item()
print(f"[weight matching] preserves model1: max|Δ| = {d_wm:.2e}")
assert d_wm < 1e-3

merger = ViTMerger(A, B, num_heads=CFG["heads"], device="cpu").eval()
d1 = (merger(x, coeff=1.0)[0] - outA).abs().max().item()
d0 = (merger(x, coeff=0.0)[0] - outB).abs().max().item()
print(f"[merger] endpoints: coeff=1 {d1:.2e} | coeff=0 {d0:.2e}")
assert d1 < 1e-3 and d0 < 1e-3
print("ALL FUNCTION-PRESERVATION CHECKS PASSED")
