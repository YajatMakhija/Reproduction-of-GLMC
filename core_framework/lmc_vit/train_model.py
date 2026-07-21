"""Train one ViT and save it to a run directory, with checkpointing/resuming."""

import argparse
import os
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from barrier import evaluate
from data import get_loaders
from models import build_vit


def train(cfg, seed, out_root, device, epochs=None, quick_batches=None, resume_dir=None):
    torch.manual_seed(seed)
    epochs = epochs if epochs is not None else cfg.get("epochs", 150)
    trainloader, testloader, num_classes = get_loaders(cfg["dataset"], batch_size=cfg.get("batch_size", 128))

    model = build_vit(cfg, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-3))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    start_epoch = 0
    best_loss = float('inf')
    out_dir = resume_dir

    if resume_dir and os.path.exists(os.path.join(resume_dir, "checkpoint.pth")):
        print(f"Resuming training from {resume_dir}")
        checkpoint = torch.load(os.path.join(resume_dir, "checkpoint.pth"), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {start_epoch-1} with best_loss {best_loss:.4f}")
    else:
        out_dir = os.path.join(out_root, f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(dict(cfg, seed=seed), f)

    for epoch in range(start_epoch, epochs):
        model.train()
        for i, (x, y) in enumerate(tqdm(trainloader, desc=f"seed{seed} ep{epoch+1}/{epochs}", leave=False)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x)[0], y).backward()
            optimizer.step()
            if quick_batches and i + 1 >= quick_batches:
                break
        scheduler.step()
        acc, loss = evaluate(model, testloader, device)
        print(f"seed{seed} ep{epoch+1}/{epochs}: test_acc={acc:.2f}% test_loss={loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': min(loss, best_loss)
        }, os.path.join(out_dir, "checkpoint.pth"))

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(out_dir, "vit.pth"))
            print(f"[*] New best model saved to {out_dir}/vit.pth")

    print(f"Finished training. Best model saved to {out_dir}/vit.pth")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/models")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume_dir", default=None, help="Directory to resume from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.seed, args.out, args.device, args.epochs, resume_dir=args.resume_dir)


if __name__ == "__main__":
    main()
