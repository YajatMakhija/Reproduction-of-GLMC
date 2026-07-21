import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess, sys as _sys
    _pkgs = ["einops", "pot", "kaggle", "pyyaml", "tqdm", "scipy", "matplotlib"]
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q",
                     "--break-system-packages", *_pkgs])
    return


@app.cell
def _():
    import os
    import sys
    import json
    import math
    import copy
    import random
    import shutil
    import tarfile
    import yaml
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    import ot

    return (
        copy,
        json,
        math,
        nn,
        np,
        optim,
        os,
        ot,
        plt,
        random,
        shutil,
        sys,
        tarfile,
        torch,
        tqdm,
        yaml,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Regularized Orthogonal Procrustes — ViT / CIFAR-10 Extension (MLRC 2026)

    This notebook reuses the reproduced `lmc_vit` codebase (weight matching +
    learned matching for ViTs) and adds our extension: **ridge-regularized**
    SVD Procrustes for the residual-stream rotation,
    `C_reg = R_B^T R_A + gamma*I`.

    We assume you already have **two independently-trained ViT checkpoints**
    (no retraining here) and run:

    1. Baselines: VAN / WM / LM (Experiment 2)
    2. The gamma sweep (Experiment 3)
    3. Eigen-angle histograms (Experiment 4)
    4. Ablation: regularize O vs. permutations vs. both (Experiment 5)
    5. Generalization note (Experiment 6 — already satisfied: this whole
       notebook *is* the ViT/CIFAR-10 domain)
    """)
    return


@app.cell
def _(os, tarfile):
    REPO_DIR = os.path.abspath("./lmc_vit")
    TAR_PATH = "./lmc_vit.tar"
    if not os.path.isfile(os.path.join(REPO_DIR, "weight_matching.py")) and os.path.exists(TAR_PATH):
        with tarfile.open(TAR_PATH) as tf:
            tf.extractall(".")
    # tar layout is lmc_vit/lmc_vit/*.py — walk down to find the actual package
    PKG_DIR = REPO_DIR
    if not os.path.isfile(os.path.join(PKG_DIR, "weight_matching.py")):
        nested = os.path.join(REPO_DIR, "lmc_vit")
        if os.path.isfile(os.path.join(nested, "weight_matching.py")):
            PKG_DIR = nested
    print("Using lmc_vit package at:", PKG_DIR)
    return (PKG_DIR,)


@app.cell
def _(PKG_DIR, sys):
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)

    from models import build_vit, reparameterize, RMSNorm
    import weight_matching as wm_mod
    from merger import ViTMerger
    from barrier import evaluate, interpolate_state_dicts, summarize, sweep_state_dict, sweep_merger
    from data import get_loaders
    from utils import project, project_to_attn_circuits, interpolate as wm_interpolate
    from enums import MatrixType

    return (
        ViTMerger,
        build_vit,
        evaluate,
        get_loaders,
        reparameterize,
        summarize,
        sweep_merger,
        sweep_state_dict,
        wm_mod,
    )


@app.cell
def _(torch):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    return (DEVICE,)


@app.cell
def _(os, shutil):
    def download_cifar10_fast(data_dir="./data", kaggle_slug="fedesoriano/cifar10-python-in-csv"):
        """Stage CIFAR-10 into <data_dir>/cifar-10-batches-py/ via the Kaggle CLI
        (much faster than the official mirror). Falls back to torchvision's own
        downloader if Kaggle creds / the slug aren't available.
        Returns True if staged successfully (so callers can skip torchvision's
        download=True and just load from disk), False otherwise.
        """
        os.makedirs(data_dir, exist_ok=True)
        marker = os.path.join(data_dir, "cifar-10-batches-py")
        if os.path.isdir(marker) and os.listdir(marker):
            print(f"CIFAR-10 already staged at {marker}")
            return True

        os.system("pip install -q kaggle --break-system-packages")
        tmp_dir = os.path.join(data_dir, "_kaggle_cifar10")
        os.makedirs(tmp_dir, exist_ok=True)
        ret = os.system(f"kaggle datasets download -d {kaggle_slug} -p {tmp_dir} --unzip")
        if ret != 0:
            print(f"kaggle CLI download failed for '{kaggle_slug}' — check credentials/slug. "
                  f"Falling back to torchvision's own download.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

        found = None
        for root, dirs, files in os.walk(tmp_dir):
            if "data_batch_1" in files:
                found = root
                break
            if "cifar-10-batches-py" in dirs:
                found = os.path.join(root, "cifar-10-batches-py")
                break
        if found is None:
            print("Kaggle download didn't contain the expected pickle batches — falling back.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

        shutil.move(found, marker)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"CIFAR-10 staged at {marker} via Kaggle")
        return True

    return (download_cifar10_fast,)


@app.cell
def _(download_cifar10_fast, get_loaders):
    DATA_ROOT = "./data"
    _staged = download_cifar10_fast(DATA_ROOT)
    trainloader, testloader, NUM_CLASSES = get_loaders(
        "CIFAR-10", batch_size=256, root=DATA_ROOT, augment=True,
        download=not _staged, num_workers=0,
    )
    print(f"train batches={len(trainloader)}  test batches={len(testloader)}  classes={NUM_CLASSES}")
    return NUM_CLASSES, testloader, trainloader


@app.cell
def _():
    # >>> edit these to match your uploaded files <
    CKPT_A, CFG_A = "/marimo/2cifar10.pth", "/marimo/config  cifar 10 2.yaml"
    CKPT_B, CFG_B = "/marimo/3cifar10.pth", "/marimo/config cifar 10 3.yaml"
    return CFG_A, CFG_B, CKPT_A, CKPT_B


@app.cell
def _(build_vit, reparameterize, torch, yaml):
    def load_trained_vit(ckpt_path, cfg_path, num_classes, device):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        m = build_vit(cfg, num_classes).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(sd)
        r = reparameterize(m, num_classes).to(device).eval()
        return r, cfg

    return (load_trained_vit,)


@app.cell
def _(CFG_A, CFG_B, CKPT_A, CKPT_B, DEVICE, NUM_CLASSES, load_trained_vit):
    model_A, cfg_A = load_trained_vit(CKPT_A, CFG_A, NUM_CLASSES, DEVICE)
    model_B, cfg_B = load_trained_vit(CKPT_B, CFG_B, NUM_CLASSES, DEVICE)
    heads = cfg_A["heads"]
    assert cfg_A["dim"] == cfg_B["dim"] and cfg_A["heads"] == cfg_B["heads"], \
        "the two models must share dim/heads to be merge-compatible"
    print(f"Loaded model A and model B — dim={cfg_A['dim']} heads={heads} depth={cfg_A['depth']}")
    return heads, model_A, model_B


@app.cell
def _(DEVICE, evaluate, model_A, model_B, testloader):
    acc_A, loss_A = evaluate(model_A, testloader, DEVICE)
    acc_B, loss_B = evaluate(model_B, testloader, DEVICE)
    print(f"Model A (reparam form): acc={acc_A:.2f}%  loss={loss_A:.4f}")
    print(f"Model B (reparam form): acc={acc_B:.2f}%  loss={loss_B:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Our Extension — Regularized Orthogonal Procrustes

    Standard weight matching solves `O_WM = argmin_O ||R_A - R_B O||_F` via
    SVD of `C = R_B^T R_A = U Σ Vᵀ`, giving `O_WM = U Vᵀ`. Directions with
    `σ_i ≈ 0` are pure floating-point noise, so `O_WM` rotates randomly
    there.

    Our fix: SVD on `C_reg = C + γI` instead. In noisy directions `γ`
    dominates, `C_reg ≈ γI`, and its SVD gives `O_reg ≈ I` (no rotation) —
    exactly the correction learned matching has to discover the hard way.
    """)
    return


@app.cell
def _(ot, torch, wm_mod):
    def compute_optimal_orthogonal_matrix_regularized(t1, t2, gamma=0.0):
        """Ridge-regularized orthogonal Procrustes.
        C = t2^T @ t1 correlates model1's (t2) residual directions with
        model0's (t1). Adding gamma*I damps low-confidence (near-zero
        singular value) directions toward the identity rotation while
        keeping the result exactly orthogonal.
        """
        C = t2.T @ t1
        d = C.shape[0]
        C_reg = C + gamma * torch.eye(d, device=C.device, dtype=C.dtype)
        U, S, Vh = torch.linalg.svd(C_reg)
        O = U @ Vh
        return O, S


    def otify_soft(cost, reg=0.05, n_iter=100):
        """Entropic-regularized OT (Sinkhorn): a *soft*, doubly-stochastic
        relaxation of the exact permutation. Used only in the Experiment 5
        ablation, to test whether softening the *permutations* (independent
        of gamma on O) also helps, or whether the gap is concentrated in O
        as the original paper's Section 6 conjecture claims.
        """
        n = cost.shape[0]
        a = torch.ones(n) / n
        b = torch.ones(n) / n
        P = ot.sinkhorn(a, b, cost, reg=reg, numItermax=n_iter)
        return P * n


    def weight_matching_regularized(model0, model1, heads, gamma=0.0, iterations=15,
                                     regularize_perm=False, perm_reg=0.05, track_O=False):
        """Same iterated weight-matching loop as weight_matching.py, but the
        residual-stream orthogonal step uses ridge-regularized Procrustes
        (gamma), and — for the ablation — the permutation steps can optionally
        use a soft Sinkhorn relaxation (regularize_perm, perm_reg) instead of
        the exact optimal-transport permutation.
        """
        dim_res = model1.pos_embedding.weight.shape[1]
        O_cum = torch.eye(dim_res, device=model1.pos_embedding.weight.device)

        for it in range(iterations):
            R0, R1 = wm_mod._residual_stack(model0, model1, include_blocks=(it > 0))
            O, _S = compute_optimal_orthogonal_matrix_regularized(R0, R1, gamma=gamma)
            O_applied = O.t()
            wm_mod.ortho_residual(model1, O_applied)
            if track_O:
                O_cum = O_applied @ O_cum

            for layer_i in range(len(model1.transformer.layers)):
                dim = model1.transformer.layers[layer_i][1].to_qkv.weight.data.shape[1]
                QK0, OUTV0 = wm_mod._head_circuits(model0, model1, heads, dim, layer_i)
                QK1, OUTV1 = wm_mod._head_circuits(model1, model1, heads, dim, layer_i)
                cost = wm_mod.get_cost_heads(QK0, QK1, heads) + wm_mod.get_cost_heads(OUTV0, OUTV1, heads)
                P = otify_soft(cost, reg=perm_reg) if regularize_perm else wm_mod.otify(cost)
                wm_mod.permute_heads(model1, P.to(QK0.device), heads, dim, layer_i)

                ff0 = torch.cat((model0.transformer.layers[layer_i][3].net[0].weight.data,
                                  model0.transformer.layers[layer_i][3].net[3].weight.data.t()), dim=1)
                ff1 = torch.cat((model1.transformer.layers[layer_i][3].net[0].weight.data,
                                  model1.transformer.layers[layer_i][3].net[3].weight.data.t()), dim=1)
                cost_ff = torch.cdist(ff0 / torch.norm(ff0, dim=-1, keepdim=True),
                                       ff1 / torch.norm(ff1, dim=-1, keepdim=True), p=1).cpu()
                P_ff = otify_soft(cost_ff, reg=perm_reg) if regularize_perm else wm_mod.otify(cost_ff)
                wm_mod.permute_mlp(model1, P_ff.to(ff0.device), layer_i)

        if track_O:
            return model1, O_cum
        return model1

    return (weight_matching_regularized,)


@app.cell
def _(math, torch):
    def eigen_angles_deg(O):
        """Eigen-angles (degrees, in [0, 360)) of an orthogonal matrix O."""
        Oc = O.detach().cpu().to(torch.complex64)
        eigvals = torch.linalg.eigvals(Oc)
        angles = torch.angle(eigvals) * 180.0 / math.pi
        angles = (angles + 360.0) % 360.0
        return angles.numpy()

    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 2 — Reproduce the Baselines (VAN / WM / LM)
    """)
    return


@app.cell
def _(
    DEVICE,
    ViTMerger,
    copy,
    heads,
    model_A,
    model_B,
    summarize,
    sweep_merger,
    sweep_state_dict,
    testloader,
    weight_matching_regularized,
):
    results_exp2 = {}

    # No alignment (VAN): interpolate the raw reparam weights directly
    van_losses, van_accs = sweep_state_dict(
        lambda: copy.deepcopy(model_A), model_A.state_dict(), model_B.state_dict(),
        testloader, DEVICE,
    )
    results_exp2["VAN"] = van_losses

    # Weight Matching (WM): gamma=0.0 == the original paper's method,
    # evaluated through an *untrained* merger (identity-initialized latents)
    a_wm, b_wm = copy.deepcopy(model_A), copy.deepcopy(model_B)
    weight_matching_regularized(a_wm, b_wm, heads, gamma=0.0, iterations=15)
    merger_wm = ViTMerger(a_wm, b_wm, num_heads=heads, device=DEVICE).to(DEVICE)
    wm_losses, wm_accs = sweep_merger(merger_wm, testloader, DEVICE)
    results_exp2["WM"] = wm_losses

    for _name in ("VAN", "WM"):
        _mb, _mid, _ = summarize(results_exp2[_name])
        print(f"{_name}: peak_barrier={_mb:.4f}  midpoint_loss={_mid:.4f}")
    return (results_exp2,)


@app.cell
def _(nn, optim, random, torch, tqdm):
    def train_learned_matching(merger, trainloader, testloader, evaluate_fn, device,
                                epochs=15, lr=1e-3, seed=41):
        torch.manual_seed(seed)
        random.seed(seed)
        optimizer = optim.Adam([p for p in merger.parameters() if p.requires_grad], lr=lr)
        criterion = nn.CrossEntropyLoss()
        sample_coeff = lambda: random.uniform(0.4, 0.6)

        for epoch in range(epochs):
            merger.train()
            for x, y in tqdm(trainloader, desc=f"LM ep{epoch+1}/{epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(merger(x, coeff=sample_coeff())[0], y).backward()
                optimizer.step()
            acc, loss = evaluate_fn(merger, testloader, device, coeff=0.5)
            print(f"[LM] epoch {epoch+1}/{epochs}: midpoint_acc={acc:.2f}% midpoint_loss={loss:.4f}")
        return merger

    return (train_learned_matching,)


@app.cell
def _(
    DEVICE,
    ViTMerger,
    copy,
    evaluate,
    heads,
    model_A,
    model_B,
    results_exp2,
    summarize,
    sweep_merger,
    testloader,
    train_learned_matching,
    trainloader,
    weight_matching_regularized,
):
    LM_EPOCHS = 15 # bump toward 15 (paper default) if you have GPU time to spare

    a_lm, b_lm = copy.deepcopy(model_A), copy.deepcopy(model_B)
    weight_matching_regularized(a_lm, b_lm, heads, gamma=0.0, iterations=15)
    merger_lm = ViTMerger(a_lm, b_lm, num_heads=heads, device=DEVICE).to(DEVICE)
    merger_lm = train_learned_matching(merger_lm, trainloader, testloader, evaluate, DEVICE, epochs=LM_EPOCHS)

    lm_losses, lm_accs = sweep_merger(merger_lm, testloader, DEVICE)
    results_exp2["LM"] = lm_losses
    _mb, _mid, _ = summarize(lm_losses)
    print(f"LM: peak_barrier={_mb:.4f}  midpoint_loss={_mid:.4f}")
    return (merger_lm,)


@app.cell
def _(json, os, results_exp2, summarize):
    def print_and_collect_table(results):
        coeffs = [round(0.1 * i, 1) for i in range(11)]
        print("\n" + "=" * 70)
        print(f"{'coeff':>6} | " + " | ".join(f"{c:>5}" for c in coeffs))
        print("-" * 70)
        rows = {}
        for name, losses in results.items():
            print(f"{name:>6} | " + " | ".join(f"{losses[c]:5.2f}" for c in coeffs))
            mb, mid, _ = summarize(losses)
            rows[name] = {"peak_barrier": mb, "midpoint_loss": mid}
        print("=" * 70)
        for name, r in rows.items():
            print(f"{name:>6}: peak_barrier={r['peak_barrier']:.4f}  midpoint={r['midpoint_loss']:.4f}")
        return rows

    exp2_summary = print_and_collect_table(results_exp2)
    os.makedirs("outputs_regularized_procrustes", exist_ok=True)
    with open("outputs_regularized_procrustes/exp2_baselines.json", "w") as f:
        json.dump({"losses": results_exp2, "summary": exp2_summary}, f, indent=2)
    return (exp2_summary,)


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 3 — The Gamma Sweep
    """)
    return


@app.cell
def _(
    DEVICE,
    ViTMerger,
    copy,
    exp2_summary,
    heads,
    model_A,
    model_B,
    summarize,
    sweep_merger,
    testloader,
    weight_matching_regularized,
):
    GAMMAS = [
        0.0,                                   # baseline WM                # below your current best
        0.001, 0.002, 0.003, 0.005,     # fine grid around 0.001
        0.01, 0.02,                      # a decade up
        0.1, 0.2,                        # where you stopped last time
        1.0,                         # confirm the U-turn (if any)
    ]
    gamma_results = {}
    gamma_table = []
    barrier_WM = exp2_summary["WM"]["peak_barrier"]
    barrier_LM = exp2_summary["LM"]["peak_barrier"]
    gap = barrier_WM - barrier_LM

    for g in GAMMAS:
        a_g, b_g = copy.deepcopy(model_A), copy.deepcopy(model_B)
        weight_matching_regularized(a_g, b_g, heads, gamma=g, iterations=15)
        merger_g = ViTMerger(a_g, b_g, num_heads=heads, device=DEVICE).to(DEVICE)
        losses_g, _ = sweep_merger(merger_g, testloader, DEVICE)
        peak_g, mid_g, _ = summarize(losses_g)
        gamma_results[g] = losses_g
        pct_closed = 100.0 * (barrier_WM - peak_g) / gap if gap > 1e-9 else float("nan")
        gamma_table.append({
            "gamma": g, "peak_barrier": peak_g, "midpoint_loss": mid_g,
            "pct_gap_closed": pct_closed,
        })
        print(f"gamma={g:<8} peak_barrier={peak_g:.4f}  %gap_closed={pct_closed:.1f}%")
    return barrier_WM, gamma_results, gamma_table


@app.cell
def _(gamma_table, plt):
    print(f"{'gamma':>10} | {'peak barrier':>12} | {'% gap closed':>12}")
    print("-" * 40)
    for row in gamma_table:
        print(f"{row['gamma']:>10} | {row['peak_barrier']:>12.4f} | {row['pct_gap_closed']:>11.1f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    gammas_plot = [r["gamma"] for r in gamma_table]
    barriers_plot = [r["peak_barrier"] for r in gamma_table]
    ax.plot(gammas_plot, barriers_plot, marker="o")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("gamma")
    ax.set_ylabel("peak loss barrier")
    ax.set_title("Regularized Procrustes: barrier vs. gamma")
    ax.grid(alpha=0.3)
    fig
    return


@app.cell
def _(barrier_WM, gamma_results, gamma_table, plt, results_exp2):
    plt.close('all')  # force a fresh figure, never reuse a cached one

    coeffs = [round(0.1 * i, 1) for i in range(11)]

    # top 2 gammas by peak_barrier, lowest (best) first
    sorted_rows = sorted(gamma_table, key=lambda r: r["peak_barrier"])
    best_row, second_row = sorted_rows[0], sorted_rows[1]
    best_gamma, second_gamma = best_row["gamma"], second_row["gamma"]

    # sanity check: fail loudly instead of silently plotting stale data
    assert best_gamma in gamma_results, f"best_gamma={best_gamma} not in gamma_results — rerun the sweep cell first"
    assert second_gamma in gamma_results, f"second_gamma={second_gamma} not in gamma_results — rerun the sweep cell first"

    print(f"#1 gamma: {best_gamma}    peak_barrier={best_row['peak_barrier']:.4f}  "
          f"({best_row['pct_gap_closed']:.1f}% of WM-LM gap closed)")
    print(f"#2 gamma: {second_gamma}  peak_barrier={second_row['peak_barrier']:.4f}  "
          f"({second_row['pct_gap_closed']:.1f}% of WM-LM gap closed)")
    print(f"WM peak_barrier={barrier_WM:.4f}")

    plt.style.use('default')
    fig2, ax2 = plt.subplots(figsize=(7, 5.5), dpi=200)

    ax2.plot(coeffs, [results_exp2["VAN"][c] for c in coeffs], "--", color="#7f7f7f",
              linewidth=1.5, label="VAN (No Alignment)")
    ax2.plot(coeffs, [results_exp2["WM"][c] for c in coeffs], "-o", color="#1f77b4",
              markersize=6, linewidth=2, label=r"WM ($\gamma=0$)")
    ax2.plot(coeffs, [gamma_results[best_gamma][c] for c in coeffs], "-^", color="#ff7f0e",
              markersize=7, linewidth=2.2, label=f"Ours ($\\gamma={best_gamma}$) — best")
    ax2.plot(coeffs, [gamma_results[second_gamma][c] for c in coeffs], "-D", color="#d62728",
              markersize=6, linewidth=1.8, linestyle=(0, (5, 2)), label=f"Ours ($\\gamma={second_gamma}$) — 2nd best")
    ax2.plot(coeffs, [results_exp2["LM"][c] for c in coeffs], "-s", color="#2ca02c",
              markersize=6, linewidth=2, label="LM (Ceiling)")

    ax2.set_xlabel(r"Interpolation Coefficient ($\lambda$)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Test Loss", fontsize=11, labelpad=8)
    ax2.set_title("Loss Landscape Along the Interpolation Path", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white", edgecolor="#d3d3d3", framealpha=1)
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')
    for spine in ax2.spines.values():
        spine.set_color('#333333')

    fig2
    return (best_gamma,)


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 4 — Eigen-Angle Analysis (reproducing Figure 5)
    """)
    return


@app.cell
def _(
    best_gamma,
    copy,
    heads,
    merger_lm,
    model_A,
    model_B,
    torch,
    weight_matching_regularized,
):
    a_e0, b_e0 = copy.deepcopy(model_A), copy.deepcopy(model_B)
    _, O_WM = weight_matching_regularized(a_e0, b_e0, heads, gamma=0.0, iterations=15, track_O=True)

    a_e1, b_e1 = copy.deepcopy(model_A), copy.deepcopy(model_B)
    _, O_reg = weight_matching_regularized(a_e1, b_e1, heads, gamma=best_gamma, iterations=15, track_O=True)

    Q_lm, _R_lm = torch.linalg.qr(merger_lm.proj["residual"].detach())
    O_LM = Q_lm
    return O_LM, O_WM, O_reg


@app.cell
def _(O_LM, math, torch):
    def eigen_angles_rad(O):
        """Eigen-angles (radians, in [0, 2*pi)) of an orthogonal matrix O."""
        Oc = O.detach().cpu().to(torch.complex64)
        eigvals = torch.linalg.eigvals(Oc)
        rads = torch.atan2(eigvals.imag, eigvals.real)
        rads = torch.where(rads < 0, rads + 2 * math.pi, rads)
        return rads.numpy()

    # paper's Fig. 5 quantities
    # merger_lm was built from a_lm/b_lm which were ALREADY WM-aligned (gamma=0)
    # before being handed to ViTMerger — so Q_lm IS the correction already, not a
    # rotation from the unaligned model. No extra composition with O_WM needed.
    O_diff = O_LM
    return O_diff, eigen_angles_rad


@app.cell
def _(np, plt):
    def plot_eigen_rose(series, labels, colors, title=None, hatches=None, num_bins=16):  # 👈 Added num_bins with a wider default
        bins = np.linspace(0, 2 * np.pi, num_bins + 1)
        full_width = (2 * np.pi) / num_bins
        bin_centers = bins[:-1] + full_width / 2

        n_series = len(series)
        pair_width = full_width * 0.95        # 👈 Increased from 0.80 to fill out empty space
        bar_width = pair_width / n_series     # each series gets a slice of that bin

        if hatches is None:
            hatches = [None] * n_series

        plt.style.use('default')
        fig = plt.figure(figsize=(6, 6.5), dpi=200)
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        for i, (rads, label, color, hatch) in enumerate(zip(series, labels, colors, hatches)):
            counts, _ = np.histogram(rads, bins=bins)
            offset = (i - (n_series - 1) / 2) * bar_width
            ax.bar(bin_centers + offset, counts, width=bar_width * 1.0, bottom=0.0,  # 👈 Changed multiplier to 1.0
                   color=color, edgecolor="#333333", linewidth=0.6,
                   hatch=hatch, alpha=0.9, label=label, zorder=3)

        ax.set_facecolor("#f9f9f9")
        fig.patch.set_facecolor("white")
        ax.set_yticklabels([])
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"], fontsize=12)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5, color="#bbbbbb")
        ax.spines['polar'].set_color('#333333')

        if title:
            ax.set_title(title, pad=24, fontsize=12, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14),
                  ncol=n_series, frameon=False, fontsize=9.5)
        fig.tight_layout()
        return fig


    return (plot_eigen_rose,)


@app.cell
def _(O_WM, O_diff, eigen_angles_rad, plot_eigen_rose):
    rads_wm = eigen_angles_rad(O_WM)
    rads_diff = eigen_angles_rad(O_diff)

    fig5 = plot_eigen_rose(
        [rads_wm, rads_diff],
        ["$O_{WM}$ (weight matching)", "$O_{diff}=O_{LM}O_{WM}^\\top$ (learned matching correction)"],
        ["#e09f67", "#8cb5db"],
        title="Eigenvalue spectra of $O_{WM}$ and $O_{diff}$",
    )
    fig5
    return


@app.cell
def _(O_WM, eigen_angles_rad, plot_eigen_rose):
    rads_wm_only = eigen_angles_rad(O_WM)

    fig_wm = plot_eigen_rose(
        [rads_wm_only],
        ["Weight matching ($O_{WM}$)"],
        ["#e8a874"],
        hatches=["."],
    )
    fig_wm
    return


@app.cell
def _(O_WM, O_reg, best_gamma, eigen_angles_rad, plot_eigen_rose):
    rads_wm1 = eigen_angles_rad(O_WM)
    rads_reg = eigen_angles_rad(O_reg)

    fig_compare = plot_eigen_rose(
        [rads_wm1, rads_reg],
        ["Weight matching ($O_{WM}$, $\\gamma=0$)", f"Ours ($O_{{reg}}$, $\\gamma={best_gamma}$)"],
        ["#e8a874", "#7fa8d4"],
        hatches=[".", None],
    )
    fig_compare
    return


@app.cell
def _(O_LM, O_WM, O_reg, best_gamma, eigen_angles_rad, plot_eigen_rose):
    rads_wm2 = eigen_angles_rad(O_WM)
    rads_reg2 = eigen_angles_rad(O_reg)
    rads_lm  = eigen_angles_rad(O_LM)   # uses the sign-canonical nearest_orthogonal(O_LM) version from before

    fig_three = plot_eigen_rose(
        [rads_wm2, rads_reg2, rads_lm],
        [
            r"WM ($\gamma=0$)",
            f"Ours ($\\gamma={best_gamma}$)",
            "LM (ceiling)",
        ],
        ["#e8a874", "#abdda4", "#7fa8d4"],   # Green/Mint for WM/Ours, Lavender for LM
        hatches=[".", "//", None],
        title=None,
    )
    fig_three
    return


@app.cell
def _(mo):
    mo.md("""
    ## Experiment 5 — Ablation: Rotation vs. Permutation
    """)
    return


@app.cell
def _(
    DEVICE,
    ViTMerger,
    best_gamma,
    copy,
    heads,
    model_A,
    model_B,
    summarize,
    sweep_merger,
    testloader,
    weight_matching_regularized,
):
    ablation_results = {}

    def run_variant(gamma, regularize_perm, perm_reg, label):
        a_v, b_v = copy.deepcopy(model_A), copy.deepcopy(model_B)
        weight_matching_regularized(a_v, b_v, heads, gamma=gamma, iterations=15,
                                     regularize_perm=regularize_perm, perm_reg=perm_reg)
        merger_v = ViTMerger(a_v, b_v, num_heads=heads, device=DEVICE).to(DEVICE)
        losses_v, _ = sweep_merger(merger_v, testloader, DEVICE)
        peak_v, mid_v, _ = summarize(losses_v)
        ablation_results[label] = {"losses": losses_v, "peak_barrier": peak_v, "midpoint_loss": mid_v}
        print(f"{label:>28}: peak_barrier={peak_v:.4f}  midpoint={mid_v:.4f}")

    run_variant(gamma=0.0,        regularize_perm=False, perm_reg=0.0,  label="(baseline) WM, gamma=0")
    run_variant(gamma=best_gamma, regularize_perm=False, perm_reg=0.0,  label="(a) regularize O only")
    run_variant(gamma=0.0,        regularize_perm=True,  perm_reg=0.05, label="(b) regularize perms only")
    run_variant(gamma=best_gamma, regularize_perm=True,  perm_reg=0.05, label="(c) regularize both")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
