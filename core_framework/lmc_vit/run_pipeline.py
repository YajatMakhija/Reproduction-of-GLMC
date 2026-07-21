"""End-to-end: train two ViTs -> weight-match + learned-match -> report barriers.

    python run_pipeline.py                 # full run (GPU)
    python run_pipeline.py --smoke         # tiny CPU-friendly correctness check
"""

import argparse
import os

import torch
import yaml

import train_model
from train_merger import train_merger
from eval_barrier import compute_barriers, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    p.add_argument("--out", default="outputs")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_epochs = 1 if args.smoke else None
    merger_epochs = 1 if args.smoke else cfg.get("merger_epochs", 15)
    quick = 5 if args.smoke else None
    wm_iters = 3 if args.smoke else 15
    max_eval = 3 if args.smoke else None

    print("=" * 20, "TRAIN MODEL 0", "=" * 20)
    f0 = train_model.train(cfg, args.seeds[0], os.path.join(args.out, "models"), args.device,
                           epochs=model_epochs, quick_batches=quick)
    print("=" * 20, "TRAIN MODEL 1", "=" * 20)
    f1 = train_model.train(cfg, args.seeds[1], os.path.join(args.out, "models"), args.device,
                           epochs=model_epochs, quick_batches=quick)

    print("=" * 20, "LEARNED MATCHING", "=" * 20)
    _, merger_dir = train_merger(f0, f1, args.device, epochs=merger_epochs, wm_iterations=wm_iters,
                                 batch_size=cfg.get("batch_size", 128),
                                 out_root=os.path.join(args.out, "mergers"), quick_batches=quick)

    print("=" * 20, "BARRIER EVALUATION", "=" * 20)
    report(compute_barriers(f0, f1, args.device, merger_dir=merger_dir,
                            wm_iterations=wm_iters, max_batches=max_eval))


if __name__ == "__main__":
    main()
