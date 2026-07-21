"""Compute and compare loss barriers: vanilla averaging (VAN), weight matching
(WM), and learned matching (LM)."""

import argparse
import copy
import os

import torch
import yaml

from barrier import sweep_merger, sweep_state_dict, summarize
from data import get_loaders
from merger import ViTMerger
from models import build_vit, reparameterize
from weight_matching import weight_matching


def _load_reparam(folder, num_classes, device):
    with open(os.path.join(folder, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    m = build_vit(cfg, num_classes).to(device)
    m.load_state_dict(torch.load(os.path.join(folder, "vit.pth"), map_location=device))
    return reparameterize(m, num_classes).to(device).eval()


def compute_barriers(folder0, folder1, device, merger_dir=None, wm_iterations=15, max_batches=None):
    with open(os.path.join(folder0, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    heads = cfg["heads"]
    _, testloader, num_classes = get_loaders(cfg["dataset"], batch_size=256, augment=False)

    r0 = _load_reparam(folder0, num_classes, device)
    r1 = _load_reparam(folder1, num_classes, device)
    results = {}

    # VAN: vanilla averaging of the (unaligned) models
    van, _ = sweep_state_dict(lambda: copy.deepcopy(r0), r0.state_dict(), r1.state_dict(),
                              testloader, device, max_batches=max_batches)
    results["VAN"] = van

    # WM: weight matching, evaluated through the untrained merger
    a, b = copy.deepcopy(r0), copy.deepcopy(r1)
    weight_matching(a, b, heads, iterations=wm_iterations)
    wm, _ = sweep_merger(ViTMerger(a, b, num_heads=heads, device=device).to(device),
                         testloader, device, max_batches=max_batches)
    results["WM"] = wm

    # LM: trained merger
    if merger_dir is not None:
        a2, b2 = copy.deepcopy(r0), copy.deepcopy(r1)
        weight_matching(a2, b2, heads, iterations=wm_iterations)
        merger = ViTMerger(a2, b2, num_heads=heads, device=device).to(device)
        merger.load_state_dict(torch.load(os.path.join(merger_dir, "merger.pth"), map_location=device))
        lm, _ = sweep_merger(merger, testloader, device, max_batches=max_batches)
        results["LM"] = lm
    return results


def report(results):
    coeffs = [round(0.1 * i, 1) for i in range(11)]
    print("\n" + "=" * 66)
    print(f"{'coeff':>6} | " + " | ".join(f"{c:>4}" for c in coeffs))
    print("-" * 66)
    for name, losses in results.items():
        print(f"{name:>6} | " + " | ".join(f"{losses[c]:4.2f}" for c in coeffs))
    print("=" * 66)
    print(f"{'method':>6} | {'max barrier':>12} | {'midpoint':>9} | {'endpoints':>9}")
    print("-" * 66)
    for name, losses in results.items():
        mb, mid, _ = summarize(losses)
        print(f"{name:>6} | {mb:>12.4f} | {mid:>9.4f} | {(losses[0.0]+losses[1.0])/2:>9.4f}")
    print("=" * 66)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder0", required=True)
    p.add_argument("--folder1", required=True)
    p.add_argument("--merger-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    report(compute_barriers(args.folder0, args.folder1, args.device, merger_dir=args.merger_dir))


if __name__ == "__main__":
    main()
