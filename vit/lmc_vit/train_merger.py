"""Weight-match two ViTs, then learn the alignment (learned matching)."""

import argparse
import os
import random
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from barrier import evaluate
from data import get_loaders
from merger import ViTMerger
from models import build_vit, reparameterize
from weight_matching import weight_matching


def load_reparam(folder, num_classes, device):
    with open(os.path.join(folder, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    m = build_vit(cfg, num_classes).to(device)
    m.load_state_dict(torch.load(os.path.join(folder, "vit.pth"), map_location=device))
    return reparameterize(m, num_classes).to(device).eval(), cfg


def train_merger(folder0, folder1, device, epochs=15, lr=1e-3, wm_iterations=15,
                 batch_size=128, out_root="outputs/mergers", quick_batches=None, seed=41):
    torch.manual_seed(seed)
    random.seed(seed)
    with open(os.path.join(folder0, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    heads = cfg["heads"]
    trainloader, testloader, num_classes = get_loaders(cfg["dataset"], batch_size=batch_size)

    m0, _ = load_reparam(folder0, num_classes, device)
    m1, _ = load_reparam(folder1, num_classes, device)

    print(f"[train_merger] weight matching ({wm_iterations} iters)...")
    weight_matching(m0, m1, heads, iterations=wm_iterations)

    merger = ViTMerger(m0, m1, num_heads=heads, device=device).to(device)
    optimizer = optim.Adam([p for p in merger.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()
    sample_coeff = lambda: random.uniform(0.4, 0.6)   # narrow-uniform interpolation coeff

    acc, loss = evaluate(merger, testloader, device, coeff=0.5)
    print(f"[train_merger] weight-matched midpoint: acc={acc:.2f}% loss={loss:.4f}")

    for epoch in range(epochs):
        merger.train()
        for i, (x, y) in enumerate(tqdm(trainloader, desc=f"merger ep{epoch+1}/{epochs}", leave=False)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(merger(x, coeff=sample_coeff())[0], y).backward()
            optimizer.step()
            if quick_batches and i + 1 >= quick_batches:
                break
        acc, loss = evaluate(merger, testloader, device, coeff=0.5)
        print(f"[train_merger] ep{epoch+1}/{epochs}: midpoint_acc={acc:.2f}% midpoint_loss={loss:.4f}")

    out_dir = os.path.join(out_root, f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(merger.state_dict(), os.path.join(out_dir, "merger.pth"))
    with open(os.path.join(out_dir, "config_merger.yaml"), "w") as f:
        yaml.safe_dump({"folder0": folder0, "folder1": folder1, "epochs": epochs,
                        "lr": lr, "wm_iterations": wm_iterations}, f)
    print(f"[train_merger] saved {out_dir}/merger.pth")
    return merger, out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder0", required=True)
    p.add_argument("--folder1", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train_merger(args.folder0, args.folder1, args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
