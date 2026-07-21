import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    return


@app.cell
def _(sys):

    import subprocess

    print("Installing required dependencies...")
    # This safely runs the equivalent of 'pip install POT' using the current Python executable
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'POT', 'einops', 'pyyaml', 'tqdm', '-q'])
    print("✅ Dependencies installed successfully!")
    return


@app.cell
def _(mo):
    mo.md("""
    # 🚀 LMC ViT — CIFAR-10 Robust Model Merging Pipeline
    """)
    return


@app.cell
def _():
    import os
    import sys
    import copy
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import marimo as mo

    # ⚙️ CONFIGURATION
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = {
        'dataset': 'CIFAR-10',
        'image_size': 32,
        'patch_size': 4,
        'dim': 256,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 512,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'dim_head': 64,
        'batch_size': 128
    }

    ROBUST_WEIGHTS = "vit (2).pth"
    STANDARD_SEEDS = ["1cifar10.pth", "2cifar10.pth", "3cifar10.pth"]

    # Ensure necessary folders exist for saving progress
    os.makedirs('outputs_cifar', exist_ok=True)
    return (
        DEVICE,
        ROBUST_WEIGHTS,
        STANDARD_SEEDS,
        cfg,
        copy,
        mo,
        nn,
        np,
        os,
        plt,
        sys,
        torch,
    )


@app.cell
def _():
    return


@app.cell
def _(DEVICE, cfg, nn, os, sys, torch):
    # 1. Add lmc_vit to python path
    if "lmc_vit" not in sys.path:
        sys.path.insert(0, "lmc_vit")

    from data import get_loaders
    from models import build_vit, reparameterize
    from weight_matching import weight_matching

    # 👈 IMPORT CLAUDE'S DUAL MERGER!
    from dual_merger import ViTMerger as DualViTMerger

    trainloader, testloader, num_classes = get_loaders(
        cfg["dataset"], 
        batch_size=cfg["batch_size"],
        root="lmc_vit/data",
        download=False
    )

    class MergerWrapper(nn.Module):
        def __init__(self, merger, lam):
            super().__init__()
            self.merger = merger
            self.lam = lam
        def forward(self, x):
            return self.merger(x, coeff=self.lam)

    def pgd_attack(model_fn, images, labels, eps=8/255, alpha=2/255, steps=7):
        images = images.clone().detach().to(DEVICE)
        labels = labels.to(DEVICE)
        images.requires_grad = True
        criterion = nn.CrossEntropyLoss()

        for _ in range(steps):
            images.requires_grad = True
            outputs = model_fn(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels)

            if images.grad is not None: images.grad.zero_()

            loss.backward()
            adv_images = images + alpha * images.grad.sign()
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)
            images = torch.clamp(images + eta, min=0, max=1).detach()

        return images

    def train_or_load_lm(model_a, model_b_wm, heads, trainloader, save_path, epochs=15):
        # 👈 INSTANTIATE THE DUAL MERGER
        merger = DualViTMerger(model_a, model_b_wm, heads, device=DEVICE).to(DEVICE)

        # Append _dual_robust to ensure it trains completely fresh
        save_path = save_path + "_dual_robust"

        if os.path.exists(save_path + ".pth"):
            print(f"Loading existing DUAL ROBUST LM from {save_path}.pth")
            merger.load_state_dict(torch.load(save_path + ".pth", map_location=DEVICE))
            return merger.eval()

        print(f"Training DUAL ROBUST LM (Saving to {save_path}.pth) ...")
        import torch.optim as optim

        # 👈 FEED BOTH SETS OF PARAMETERS TO THE OPTIMIZER!
        params = list(merger.proj_A.parameters()) + list(merger.proj_B.parameters())
        optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for ep in range(epochs):
            total_clean_loss, total_rob_loss = 0, 0
            merger.train()

            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # 1. Generate Attack Images safely
                merger.eval() 
                adv_images = pgd_attack(lambda x: merger(x, coeff=0.5), images, labels)
                merger.train() 

                # 2. Clear junk attacker gradients!
                optimizer.zero_grad()

                # 3. Compute Clean Loss
                out_c = merger(images, coeff=0.5)
                if isinstance(out_c, tuple): out_c = out_c[0]
                loss_c = criterion(out_c, labels)

                # 4. Compute Robust Loss
                out_r = merger(adv_images, coeff=0.5)
                if isinstance(out_r, tuple): out_r = out_r[0]
                loss_r = criterion(out_r, labels)

                # 5. Joint Optimization
                # 5. Joint Optimization
                loss = loss_c + (2.0 * loss_r)
                loss.backward()
                optimizer.step()

                total_clean_loss += loss_c.item()
                total_rob_loss += loss_r.item()

            print(f"  Epoch {ep+1}/{epochs} | Clean Loss: {total_clean_loss/len(trainloader):.4f} | Rob Loss: {total_rob_loss/len(trainloader):.4f}")

        torch.save(merger.state_dict(), save_path + ".pth")
        return merger.eval()

    def evaluate_model(model_fn, loader, is_lm=False, lam=None):
        criterion = nn.CrossEntropyLoss()
        clean_loss, rob_loss = 0.0, 0.0
        clean_corr, rob_corr = 0, 0
        total = 0

        if is_lm:
            pgd_model = MergerWrapper(model_fn, lam)
            eval_fn = lambda x: model_fn(x, coeff=lam)
        else:
            pgd_model = model_fn
            eval_fn = model_fn

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                out_c = eval_fn(images)
                if isinstance(out_c, tuple): out_c = out_c[0]
                clean_loss += criterion(out_c, labels).item() * labels.size(0)
                clean_corr += (out_c.argmax(1) == labels).sum().item()

            adv_images = pgd_attack(pgd_model, images, labels)
            with torch.no_grad():
                out_r = eval_fn(adv_images)
                if isinstance(out_r, tuple): out_r = out_r[0]
                rob_loss += criterion(out_r, labels).item() * labels.size(0)
                rob_corr += (out_r.argmax(1) == labels).sum().item()

            total += labels.size(0)

        return {
            "clean_loss": clean_loss / total,
            "rob_loss": rob_loss / total,
            "clean_acc": (clean_corr / total) * 100,
            "rob_acc": (rob_corr / total) * 100
        }

    return (
        build_vit,
        evaluate_model,
        num_classes,
        reparameterize,
        testloader,
        train_or_load_lm,
        trainloader,
        weight_matching,
    )


@app.cell
def _(os):


    # PyTorch STRICTLY requires the folder to be named 'cifar-10-batches-py'
    data_dir = "lmc_vit/data/cifar-10-batches-py"
    os.makedirs(data_dir, exist_ok=True)

    print("Installing Kaggle...")
    os.system("pip install -q kaggle")

    print("Downloading CIFAR-10 via Kaggle (Lightning Fast!)...")
    os.system(f"kaggle datasets download -d pankrzysiu/cifar10-python -p {data_dir} --unzip")
    return


@app.cell
def _():
    return


@app.cell
def _(
    DEVICE,
    ROBUST_WEIGHTS,
    build_vit,
    cfg,
    num_classes,
    reparameterize,
    torch,
):
    print("Loading Robust Model A...")
    model_a_base = build_vit(cfg, num_classes).to(DEVICE)
    model_a_base.load_state_dict(torch.load(ROBUST_WEIGHTS, map_location=DEVICE, weights_only=False))
    model_a = reparameterize(model_a_base, num_classes).to(DEVICE).eval()
    print("✅ Model A ready.")
    return model_a, model_a_base


@app.cell
def _():
    return


@app.cell
def _(
    DEVICE,
    STANDARD_SEEDS,
    copy,
    model_a_base,
    num_classes,
    reparameterize,
    torch,
):
    print("Validating Standard Models to prevent shape mismatches...")
    valid_seeds = []
    standard_models = {}

    for _seed in STANDARD_SEEDS:
        try:
            # Deepcopy guarantees EXACT architectural match to model_a_base!
            _base = copy.deepcopy(model_a_base)
            _base.load_state_dict(torch.load(_seed, map_location=DEVICE, weights_only=False))
            _model_b = reparameterize(_base, num_classes).to(DEVICE).eval()
            standard_models[_seed] = _model_b
            valid_seeds.append(_seed)
            print(f"✅ {_seed} loaded successfully.")
        except Exception as e:
            print(f"❌ Skipping {_seed} due to incompatible weights shape: {e}")

    print(f"Valid seeds for processing: {valid_seeds}")
    return standard_models, valid_seeds


@app.cell
def _():
    return


@app.cell
def _(
    copy,
    evaluate_model,
    model_a,
    np,
    os,
    standard_models,
    testloader,
    torch,
    valid_seeds,
):
    print("🔄 Running VANILLA AVERAGING...")
    vanilla_results_path = os.path.join("outputs_cifar", "vanilla_results.pth")

    if os.path.exists(vanilla_results_path):
        # 👈 FIX: Added weights_only=False
        vanilla_results = torch.load(vanilla_results_path, map_location='cpu', weights_only=False)
        print("✅ Loaded previous Vanilla results from disk.")
    else:
        vanilla_results = {}
        _lambdas = np.linspace(0, 1, 11)

        for _seed in valid_seeds:
            print(f"Evaluating Vanilla for {_seed}...")
            vanilla_results[_seed] = {}
            _model_b = standard_models[_seed]

            for _lam in _lambdas:
                _m_interp = copy.deepcopy(model_a)
                _s_interp = _m_interp.state_dict()
                for _k in _s_interp:
                    _s_interp[_k] = _lam * model_a.state_dict()[_k] + (1 - _lam) * _model_b.state_dict()[_k]
                _m_interp.load_state_dict(_s_interp)
                _m_interp.eval()

                _res = evaluate_model(_m_interp, testloader)
                vanilla_results[_seed][round(_lam, 1)] = _res
                print(f"  lam={_lam:.1f} | Clean Acc: {_res['clean_acc']:.1f}% | Clean Loss: {_res['clean_loss']:.4f}")

        torch.save(vanilla_results, vanilla_results_path)
        print("✅ Saved Vanilla results to .pth file!")
    return (vanilla_results,)


@app.cell
def _():
    return


@app.cell
def _(
    cfg,
    copy,
    evaluate_model,
    model_a,
    np,
    os,
    standard_models,
    testloader,
    torch,
    valid_seeds,
    weight_matching,
):
    print("🔄 Running WEIGHT MATCHING...")
    wm_results_path = os.path.join("outputs_cifar", "wm_results.pth")
    aligned_wm_models = {}

    if os.path.exists(wm_results_path):
        # 👈 FIX: Added weights_only=False
        wm_results = torch.load(wm_results_path, map_location='cpu', weights_only=False)
        print("✅ Loaded previous WM results from disk.")

        for _seed in valid_seeds:
            try:
                _m_wm = copy.deepcopy(standard_models[_seed])
                weight_matching(model_a, _m_wm, cfg['heads'], iterations=15)
                aligned_wm_models[_seed] = _m_wm
            except: pass
    else:
        wm_results = {}
        _lambdas = np.linspace(0, 1, 11)

        for _seed in valid_seeds:
            print(f"Evaluating Weight Matching for {_seed}...")
            _m_wm = copy.deepcopy(standard_models[_seed])

            try:
                weight_matching(model_a, _m_wm, cfg['heads'], iterations=15)
                aligned_wm_models[_seed] = _m_wm
            except Exception as e:
                print(f"❌ Weight matching failed for {_seed}: {e}")
                continue

            wm_results[_seed] = {}
            for _lam in _lambdas:
                _m_interp = copy.deepcopy(model_a)
                _s_interp = _m_interp.state_dict()
                for _k in _s_interp:
                    _s_interp[_k] = _lam * model_a.state_dict()[_k] + (1 - _lam) * _m_wm.state_dict()[_k]
                _m_interp.load_state_dict(_s_interp)
                _m_interp.eval()

                _res = evaluate_model(_m_interp, testloader)
                wm_results[_seed][round(_lam, 1)] = _res
                print(f"  lam={_lam:.1f} | Clean Acc: {_res['clean_acc']:.1f}% | Clean Loss: {_res['clean_loss']:.4f}")

        torch.save(wm_results, wm_results_path)
        print("✅ Saved WM results to .pth file!")
    return aligned_wm_models, wm_results


@app.cell
def _():
    return


@app.cell
def _(
    aligned_wm_models,
    cfg,
    evaluate_model,
    model_a,
    np,
    os,
    testloader,
    torch,
    train_or_load_lm,
    trainloader,
):
    print("🔄 Running DUAL ROBUST-AWARE LEARNED MATCHING...")

    # Save to a completely new file to ensure it doesn't accidentally load old results!
    lm_results_path = os.path.join("outputs_cifar", "lm_results_dual_robust.pth")

    if os.path.exists(lm_results_path):
        lm_results = torch.load(lm_results_path, map_location='cpu', weights_only=False)
        print("✅ Loaded previous DUAL ROBUST LM results from disk.")
    else:
        lm_results = {}
        _lambdas = np.linspace(0, 1, 11)

        for _seed, _m_wm in aligned_wm_models.items():
            print(f"Evaluating Dual Robust Learned Matching for {_seed}...")
            _save_path = os.path.join("outputs_cifar", f"lm_merger_{os.path.basename(_seed)}")

            # This triggers the new DualViTMerger
            _merger_lm = train_or_load_lm(model_a, _m_wm, cfg['heads'], trainloader, _save_path, epochs=15)

            lm_results[_seed] = {}
            for _lam in _lambdas:
                _res = evaluate_model(_merger_lm, testloader, is_lm=True, lam=_lam)
                lm_results[_seed][round(_lam, 1)] = _res
                print(f"  lam={_lam:.1f} | Clean Acc: {_res['clean_acc']:.1f}% | Clean Loss: {_res['clean_loss']:.4f}")

        torch.save(lm_results, lm_results_path)
        print("✅ Saved DUAL ROBUST LM results to .pth file!")
    return (lm_results,)


@app.cell
def _():
    return


@app.cell
def _(lm_results, np, plt, valid_seeds, vanilla_results, wm_results):
    def plot_aggregated_barriers():
        if not valid_seeds:
            print("No valid seeds found.")
            return

        _lams = np.linspace(0, 1, 11)

        # Helper to extract all seeds for a specific method and key
        def get_all_seeds_data(res_dict, key):
            # Returns shape: (num_seeds, num_lambdas)
            all_data = []
            for _s in valid_seeds:
                if _s in res_dict:
                    all_data.append([res_dict[_s][round(_lam, 1)][key] for _lam in _lams])
            return np.array(all_data)

        # Helper to calculate average barrier across all seeds
        def calc_avg_barrier(data_matrix, is_loss=True):
            # data_matrix is (num_seeds, num_lambdas)
            barriers = []
            for i in range(len(data_matrix)):
                vals = data_matrix[i]
                mid = vals[5]  # Index 5 is lam=0.5
                avg = (vals[0] + vals[-1]) / 2.0
                bar = (mid - avg) if is_loss else (avg - mid)
                barriers.append(bar)
            return np.mean(barriers), np.std(barriers)

        # Extract matrices
        van_c_loss = get_all_seeds_data(vanilla_results, "clean_loss")
        van_r_loss = get_all_seeds_data(vanilla_results, "rob_loss")
        wm_c_loss = get_all_seeds_data(wm_results, "clean_loss")
        wm_r_loss = get_all_seeds_data(wm_results, "rob_loss")
        lm_c_loss = get_all_seeds_data(lm_results, "clean_loss")
        lm_r_loss = get_all_seeds_data(lm_results, "rob_loss")

        van_c_acc = get_all_seeds_data(vanilla_results, "clean_acc")
        van_r_acc = get_all_seeds_data(vanilla_results, "rob_acc")
        wm_c_acc = get_all_seeds_data(wm_results, "clean_acc")
        wm_r_acc = get_all_seeds_data(wm_results, "rob_acc")
        lm_c_acc = get_all_seeds_data(lm_results, "clean_acc")
        lm_r_acc = get_all_seeds_data(lm_results, "rob_acc")

        def print_stats(name, matrix, is_loss):
            mean_b, std_b = calc_avg_barrier(matrix, is_loss)
            unit = "" if is_loss else "%"
            print(f"{name:20s}: {mean_b:>7.4f} ± {std_b:.4f}{unit}")

        print("\n" + "="*60)
        print("📊 AVERAGE LOSS BARRIERS (across all seeds)")
        print("="*60)
        print_stats("Vanilla Clean Loss", van_c_loss, True)
        print_stats("Vanilla Robust Loss", van_r_loss, True)
        print_stats("WM Clean Loss", wm_c_loss, True)
        print_stats("WM Robust Loss", wm_r_loss, True)
        print_stats("LM Clean Loss", lm_c_loss, True)
        print_stats("LM Robust Loss", lm_r_loss, True)

        print("\n" + "="*60)
        print("📈 AVERAGE ACCURACY DROPS (across all seeds)")
        print("="*60)
        print_stats("Vanilla Clean Acc", van_c_acc, False)
        print_stats("Vanilla Robust Acc", van_r_acc, False)
        print_stats("WM Clean Acc", wm_c_acc, False)
        print_stats("WM Robust Acc", wm_r_acc, False)
        print_stats("LM Clean Acc", lm_c_acc, False)
        print_stats("LM Robust Acc", lm_r_acc, False)

        # Plotting Helper for Shaded Error Bands
        def plot_with_std(ax, x, matrix, color, marker, label):
            mean = np.mean(matrix, axis=0)
            std = np.std(matrix, axis=0)
            ax.plot(x, mean, color=color, marker=marker, label=label)
            # Shade the standard deviation area!
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        # Plot Loss Curves
        plt.style.use('default')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        plot_with_std(axes[0], _lams, van_c_loss, '#2ca02c', 'd', 'Vanilla')
        plot_with_std(axes[0], _lams, wm_c_loss, '#ff7f0e', 'v', 'Weight Matching')
        plot_with_std(axes[0], _lams, lm_c_loss, '#1f77b4', 'o', 'Learned Matching')
        axes[0].set_title(f"Clean Loss (Avg over {len(valid_seeds)} seeds)")
        axes[0].set_xlabel(r"Interpolation coefficient ($\lambda$)")
        axes[0].set_ylabel("Loss $\downarrow$")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        plot_with_std(axes[1], _lams, van_r_loss, '#2ca02c', 'd', 'Vanilla')
        plot_with_std(axes[1], _lams, wm_r_loss, '#ff7f0e', 'v', 'Weight Matching')
        plot_with_std(axes[1], _lams, lm_r_loss, '#1f77b4', 'o', 'Learned Matching')
        axes[1].set_title(f"Robust Loss (Avg over {len(valid_seeds)} seeds)")
        axes[1].set_xlabel(r"Interpolation coefficient ($\lambda$)")
        axes[1].set_ylabel("Loss $\downarrow$")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        import os
        os.makedirs("outputs_cifar", exist_ok=True)
        plt.savefig(f"outputs_cifar/loss_barriers_aggregated.pdf", dpi=300)
        plt.show()

    plot_aggregated_barriers()
    return


@app.cell
def _():
    return


@app.cell
def _(lm_results, np, os, plt, valid_seeds, vanilla_results, wm_results):
    def plot_additional_metrics():
        if not valid_seeds:
            print("No valid seeds to plot.")
            return
        
        _lams = np.linspace(0, 1, 11)
    
        # Helper to safely extract data
        def get_all_seeds_data(res_dict, key):
            all_data = []
            for _s in valid_seeds:
                if _s in res_dict:
                    all_data.append([res_dict[_s][round(_lam, 1)][key] for _lam in _lams])
            return np.array(all_data)

        # 1. Extract Accuracy Data safely
        van_c_acc = get_all_seeds_data(vanilla_results, "clean_acc")
        van_r_acc = get_all_seeds_data(vanilla_results, "rob_acc")
        wm_c_acc = get_all_seeds_data(wm_results, "clean_acc")
        wm_r_acc = get_all_seeds_data(wm_results, "rob_acc")
        lm_c_acc = get_all_seeds_data(lm_results, "clean_acc")
        lm_r_acc = get_all_seeds_data(lm_results, "rob_acc")

        def plot_with_std(ax, x, matrix, color, marker, label):
            mean = np.mean(matrix, axis=0)
            std = np.std(matrix, axis=0)
            ax.plot(x, mean, color=color, marker=marker, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        # ==========================================
        # FIGURE 1: Accuracy Paths (The Mountains)
        # ==========================================
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
        plot_with_std(axes1[0], _lams, van_c_acc, '#2ca02c', 'd', 'Vanilla')
        plot_with_std(axes1[0], _lams, wm_c_acc, '#ff7f0e', 'v', 'Weight Matching')
        plot_with_std(axes1[0], _lams, lm_c_acc, '#1f77b4', 'o', 'Learned Matching (Dual)')
        axes1[0].set_title(f"Clean Accuracy (Avg over {len(valid_seeds)} seeds)")
        axes1[0].set_xlabel(r"Interpolation coefficient ($\lambda$)")
        axes1[0].set_ylabel(r"Accuracy (%) $\uparrow$")
        axes1[0].grid(True, alpha=0.3)
        axes1[0].legend()

        plot_with_std(axes1[1], _lams, van_r_acc, '#2ca02c', 'd', 'Vanilla')
        plot_with_std(axes1[1], _lams, wm_r_acc, '#ff7f0e', 'v', 'Weight Matching')
        plot_with_std(axes1[1], _lams, lm_r_acc, '#1f77b4', 'o', 'Learned Matching (Dual)')
        axes1[1].set_title(f"Robust Accuracy (Avg over {len(valid_seeds)} seeds)")
        axes1[1].set_xlabel(r"Interpolation coefficient ($\lambda$)")
        axes1[1].set_ylabel(r"Accuracy (%) $\uparrow$")
        axes1[1].grid(True, alpha=0.3)
        axes1[1].legend()
    
        plt.tight_layout()
        os.makedirs("outputs_cifar", exist_ok=True)
        plt.savefig(f"outputs_cifar/accuracy_paths.pdf", dpi=300)
        plt.show()

        # ==========================================
        # FIGURE 2: The Pareto Trade-off Curve
        # ==========================================
        fig2, ax2 = plt.subplots(figsize=(9, 6))
    
        # Calculate means
        mean_van_c = np.mean(van_c_acc, axis=0)
        mean_van_r = np.mean(van_r_acc, axis=0)
        mean_wm_c = np.mean(wm_c_acc, axis=0)
        mean_wm_r = np.mean(wm_r_acc, axis=0)
        mean_lm_c = np.mean(lm_c_acc, axis=0)
        mean_lm_r = np.mean(lm_r_acc, axis=0)

        # Plot curves
        ax2.plot(mean_van_c, mean_van_r, marker='d', color='#2ca02c', label='Vanilla Path', linestyle='--', alpha=0.6)
        ax2.plot(mean_wm_c, mean_wm_r, marker='v', color='#ff7f0e', label='WM Path', linestyle='--', alpha=0.6)
        ax2.plot(mean_lm_c, mean_lm_r, marker='o', color='#1f77b4', label='LM Path (Ours)', linewidth=2.5)
    
        # Highlight endpoints
        ax2.scatter(mean_van_c[0], mean_van_r[0], color='black', s=100, zorder=5, label='Endpoints ($\lambda=0, 1$)')
        ax2.scatter(mean_van_c[-1], mean_van_r[-1], color='black', s=100, zorder=5)

        ax2.set_title("Clean vs. Robust Accuracy Trade-off (Pareto Front)")
        ax2.set_xlabel("Clean Accuracy (%)")
        ax2.set_ylabel("Robust Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
        plt.tight_layout()
        plt.savefig(f"outputs_cifar/pareto_tradeoff.pdf", dpi=300)
        plt.show()

    plot_additional_metrics()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
