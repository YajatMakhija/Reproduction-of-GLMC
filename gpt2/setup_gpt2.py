import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os, sys, subprocess, tarfile
    import marimo as mo

    # 1. Install dependencies (added kaggle)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'POT', 'einops', 'pyyaml', 'tqdm', 'matplotlib', 'scipy', 'kaggle', '-q'])

    # 2. Extract lmc_vit repository
    tar_path = "lmc_vit.tar"
    work_dir = "lmc_vit"
    if not os.path.exists(work_dir):
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=".")
    sys.path.insert(0, work_dir)

    # 3. Download CIFAR-10 via Kaggle
    data_dir = "./data"  # PyTorch's default data folder
    os.makedirs(data_dir, exist_ok=True)

    # Download and unzip directly to the data directory
    print("Downloading CIFAR-10 via Kaggle...")
    os.system(f"kaggle datasets download -d pankrzysiu/cifar10-python -p {data_dir} --unzip")

    # Return Marimo markdown to display the result
    mo.md(f"✅ Setup complete! Files now in `{data_dir}`: {os.listdir(data_dir)}")
    return (os,)


@app.cell
def _():
    # Cell 2 — Configuration
    MODEL_0_CONFIG = "config1.yaml"
    MODEL_0_WEIGHTS = "1cifar10.pth"

    MODEL_1_CONFIG = "config2.yaml"
    MODEL_1_WEIGHTS = "2cifar10.pth"

    DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    LM_EPOCHS = 15
    WM_ITERATIONS = 15
    LM_LR = 1e-3
    BATCH_SIZE = 128
    print(f"Device: {DEVICE}")
    return (
        BATCH_SIZE,
        DEVICE,
        LM_EPOCHS,
        LM_LR,
        MODEL_0_CONFIG,
        MODEL_0_WEIGHTS,
        MODEL_1_CONFIG,
        MODEL_1_WEIGHTS,
        WM_ITERATIONS,
    )


@app.cell
def _(BATCH_SIZE, WM_ITERATIONS):
    import copy, random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import yaml
    from tqdm import tqdm
    from data import get_loaders
    from models import build_vit, reparameterize
    from weight_matching import weight_matching
    from merger import ViTMerger
    from barrier import evaluate, sweep_merger, summarize

    def load_reparam(config_path, weights_path, num_classes, device):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        m = build_vit(cfg, num_classes).to(device)
        m.load_state_dict(torch.load(weights_path, map_location=device))
        return reparameterize(m, num_classes).to(device).eval(), cfg

    def run_lm_with_sampling(cfg0, w0, cfg1, w1, sample_fn, label, device, epochs, lr):
        """
        Run the full LM pipeline with a specific coefficient sampling function.
        """
        with open(cfg0) as f:
            cfg = yaml.safe_load(f)
        heads = cfg["heads"]
        trainloader, testloader, num_classes = get_loaders(cfg["dataset"], batch_size=BATCH_SIZE)
    
        # -------------------------------------------------------------------
        # FIX FOR BROKEN PIPE: Disable background workers in notebook
        # -------------------------------------------------------------------
        trainloader = torch.utils.data.DataLoader(
            trainloader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        testloader = torch.utils.data.DataLoader(
            testloader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        # -------------------------------------------------------------------
    
        m0, _ = load_reparam(cfg0, w0, num_classes, device)
        m1, _ = load_reparam(cfg1, w1, num_classes, device)
    
        # Weight match first
        weight_matching(m0, m1, heads, iterations=WM_ITERATIONS)
        merger = ViTMerger(m0, m1, num_heads=heads, device=device).to(device)
    
        optimizer = optim.Adam([p for p in merger.parameters() if p.requires_grad], lr=lr)
        criterion = nn.CrossEntropyLoss()
    
        train_losses = []
        test_barriers = []
    
        # Pre-training barrier
        merger.eval()
        init_losses, _ = sweep_merger(merger, testloader, device)
        mb_init, _, _ = summarize(init_losses)
        test_barriers.append(mb_init)
    
        for epoch in range(epochs):
            merger.train()
            epoch_loss, n = 0.0, 0
            for x, y in tqdm(trainloader, desc=f"{label} ep{epoch+1}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                coeff = sample_fn()  # <-- THIS IS THE VARIABLE UNDER TEST
                loss = criterion(merger(x, coeff=coeff)[0], y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            
            train_losses.append(epoch_loss / n)
            merger.eval()
            test_losses, _ = sweep_merger(merger, testloader, device)
            mb, mid, _ = summarize(test_losses)
            test_barriers.append(mb)
            print(f"  {label} Epoch {epoch+1}/{epochs}: train={train_losses[-1]:.4f}, barrier={mb:.4f}")
        
        # Final full sweep for plotting
        final_losses, final_accs = sweep_merger(merger, testloader, device)
        return train_losses, test_barriers, final_losses, final_accs

    print("✅ Helper function defined (with multiprocessing fix)")
    return random, run_lm_with_sampling, torch


@app.cell
def _(os):
    os.system("nvidia-smi")

    return


@app.cell
def _(os):
    print("Uninstalling old PyTorch...")
    os.system("pip uninstall -y torch torchvision torchaudio")
    print("Installing GPU PyTorch (this is a large download)...")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("Done!")

    return


@app.cell
def _(torch):

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    return


@app.cell
def _(
    DEVICE,
    LM_EPOCHS,
    LM_LR,
    MODEL_0_CONFIG,
    MODEL_0_WEIGHTS,
    MODEL_1_CONFIG,
    MODEL_1_WEIGHTS,
    random,
    run_lm_with_sampling,
):
    # ============================================================
    # CONDITION 1: Narrow sampling — λ ~ Uniform(0.4, 0.6)
    # This is the paper's default (Algorithm 1, line: sample_coeff)
    # ============================================================
    print("=" * 50)
    print("CONDITION 1: λ ~ Uniform(0.4, 0.6) [Paper Default]")
    print("=" * 50)
    narrow_train, narrow_barriers, narrow_final_losses, narrow_final_accs = run_lm_with_sampling(
        MODEL_0_CONFIG, MODEL_0_WEIGHTS, MODEL_1_CONFIG, MODEL_1_WEIGHTS,
        sample_fn=lambda: random.uniform(0.4, 0.6),
        label="Narrow",
        device=DEVICE, epochs=LM_EPOCHS, lr=LM_LR
    )

    # ============================================================
    # CONDITION 2: Broad sampling — λ ~ Uniform(0.0, 1.0)
    # Paper Appendix C.1.1: broader range means endpoints also
    # get optimized, but midpoint may be slightly worse.
    # ============================================================
    print("\n" + "=" * 50)
    print("CONDITION 2: λ ~ Uniform(0.0, 1.0) [Broad]")
    print("=" * 50)
    broad_train, broad_barriers, broad_final_losses, broad_final_accs = run_lm_with_sampling(
        MODEL_0_CONFIG, MODEL_0_WEIGHTS, MODEL_1_CONFIG, MODEL_1_WEIGHTS,
        sample_fn=lambda: random.uniform(0.0, 1.0),
        label="Broad",
        device=DEVICE, epochs=LM_EPOCHS, lr=LM_LR
    )
    return (
        broad_barriers,
        broad_final_accs,
        broad_final_losses,
        broad_train,
        narrow_barriers,
        narrow_final_accs,
        narrow_final_losses,
        narrow_train,
    )


@app.cell
def _(
    broad_barriers,
    broad_final_accs,
    broad_final_losses,
    broad_train,
    narrow_barriers,
    narrow_final_accs,
    narrow_final_losses,
    narrow_train,
    torch,
):

    # Package all the variables from Cell 4 into a dictionary
    backup_data = {
        "narrow_train": narrow_train,
        "narrow_barriers": narrow_barriers,
        "narrow_final_losses": narrow_final_losses,
        "narrow_final_accs": narrow_final_accs,
        "broad_train": broad_train,
        "broad_barriers": broad_barriers,
        "broad_final_losses": broad_final_losses,
        "broad_final_accs": broad_final_accs,
    }

    # Save the dictionary to your hard drive
    torch.save(backup_data, "exp8_backup.pt")
    print("✅ Data safely backed up to 'exp8_backup.pt'")
    return


@app.cell
def _(
    DEVICE,
    LM_EPOCHS,
    LM_LR,
    MODEL_0_CONFIG,
    MODEL_0_WEIGHTS,
    MODEL_1_CONFIG,
    MODEL_1_WEIGHTS,
    random,
    run_lm_with_sampling,
):

    # ============================================================
    # CONDITION 3: Gaussian sampling N(0.5, 0.1)
    # ============================================================
    print("\n" + "=" * 50)
    print("CONDITION 3: λ ~ N(0.5, 0.1) [Gaussian]")
    print("=" * 50)
    gaussian_train, gaussian_barriers, gaussian_final_losses, gaussian_final_accs = run_lm_with_sampling(
        MODEL_0_CONFIG, MODEL_0_WEIGHTS, MODEL_1_CONFIG, MODEL_1_WEIGHTS,
        sample_fn=lambda: random.gauss(0.5, 0.1),
        label="Gaussian",
        device=DEVICE, epochs=LM_EPOCHS, lr=LM_LR
    )

    # ============================================================
    # CONDITION 4: Fixed interpolation (α = 0.5)
    # ============================================================
    print("\n" + "=" * 50)
    print("CONDITION 4: Fixed interpolation (α = 0.5)")
    print("=" * 50)
    fixed_train, fixed_barriers, fixed_final_losses, fixed_final_accs = run_lm_with_sampling(
        MODEL_0_CONFIG, MODEL_0_WEIGHTS, MODEL_1_CONFIG, MODEL_1_WEIGHTS,
        sample_fn=lambda: 0.5,
        label="Fixed",
        device=DEVICE, epochs=LM_EPOCHS, lr=LM_LR
    )
    return (
        fixed_barriers,
        fixed_final_accs,
        fixed_final_losses,
        fixed_train,
        gaussian_barriers,
        gaussian_final_accs,
        gaussian_final_losses,
        gaussian_train,
    )


@app.cell
def _(
    broad_barriers,
    broad_final_accs,
    broad_final_losses,
    broad_train,
    fixed_barriers,
    fixed_final_accs,
    fixed_final_losses,
    fixed_train,
    gaussian_barriers,
    gaussian_final_accs,
    gaussian_final_losses,
    gaussian_train,
    narrow_barriers,
    narrow_final_accs,
    narrow_final_losses,
    narrow_train,
    torch,
):


    # Package all the variables from all 4 conditions into a dictionary
    backup_data_full = {
        # Original conditions
        "narrow_train": narrow_train,
        "narrow_barriers": narrow_barriers,
        "narrow_final_losses": narrow_final_losses,
        "narrow_final_accs": narrow_final_accs,
    
        "broad_train": broad_train,
        "broad_barriers": broad_barriers,
        "broad_final_losses": broad_final_losses,
        "broad_final_accs": broad_final_accs,
    
        # New conditions
        "gaussian_train": gaussian_train,
        "gaussian_barriers": gaussian_barriers,
        "gaussian_final_losses": gaussian_final_losses,
        "gaussian_final_accs": gaussian_final_accs,
    
        "fixed_train": fixed_train,
        "fixed_barriers": fixed_barriers,
        "fixed_final_losses": fixed_final_losses,
        "fixed_final_accs": fixed_final_accs,
    }

    # Save the dictionary to your hard drive
    torch.save(backup_data_full, "exp8_backup_COMPLETE.pt")
    print("✅ All 4 conditions safely backed up to 'exp8_backup_COMPLETE.pt'")
    return


@app.cell
def _(
    broad_final_losses,
    fixed_final_losses,
    gaussian_final_losses,
    narrow_final_losses,
):
    import matplotlib.pyplot as plt

    # Extract data for all 4 conditions
    coeffs = sorted(narrow_final_losses.keys())
    narrow_y = [narrow_final_losses[c] for c in coeffs]
    broad_y = [broad_final_losses[c] for c in coeffs]
    gaussian_y = [gaussian_final_losses[c] for c in coeffs]
    fixed_y = [fixed_final_losses[c] for c in coeffs]

    # Calculate exact min and max for dynamic scaling so nothing is cut off
    all_y = narrow_y + broad_y + gaussian_y + fixed_y
    min_y = min(all_y)
    max_y = max(all_y)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor('#EAEAF2')
    ax.grid(color='white', linestyle='-', linewidth=1.5)
    ax.set_axisbelow(True)

    # 1. Gaussian: Blue, Circle markers
    plt.plot(coeffs, gaussian_y, color='tab:blue', marker='o', linestyle='-', linewidth=2, markersize=6, label=r'Gaussian sampling $\mathcal{N}(0.5, 0.1)$')

    # 2. Uniform [0.4, 0.6]: Orange, Square markers
    plt.plot(coeffs, narrow_y, color='tab:orange', marker='s', linestyle='-', linewidth=2, markersize=6, label='Uniform sampling [0.4, 0.6]')

    # 3. Uniform [0.0, 1.0]: Green, Diamond markers
    plt.plot(coeffs, broad_y, color='tab:green', marker='D', linestyle='-', linewidth=2, markersize=6, label='Uniform sampling [0.0, 1.0]')

    # 4. Fixed (alpha=0.5): Red, Triangle markers
    plt.plot(coeffs, fixed_y, color='tab:red', marker='^', linestyle='-', linewidth=2, markersize=6, label=r'Fixed interpolation ($\alpha=0.5$)')

    # Fixed scaling padding
    plt.ylim(min_y - 0.02, max_y + 0.02)
    plt.xlim(-0.05, 1.05)

    plt.ylabel(r'Loss $\downarrow$', fontsize=13)
    plt.xlabel(r'Interpolation coefficient ($\lambda$)', fontsize=13)
    plt.xticks([0.0, 1.0], ['Model A', r'$\pi$(Model B)'], fontsize=12)

    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='lightgrey', fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig("paper_figure_6_COMPLETE.png", dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(
    broad_final_losses,
    fixed_final_losses,
    gaussian_final_losses,
    narrow_final_losses,
):
    import numpy as np
    import pandas as pd

    def calculate_theoretical_metrics(loss_dict):
        coeffs = sorted(loss_dict.keys())
        losses = [loss_dict[c] for c in coeffs]
    
        # 1. Midpoint Loss (λ=0.5)
        midpoint = loss_dict[0.5]
    
        # 2. Max Loss Barrier (Max deviation from the linear baseline)
        loss_a, loss_b = losses[0], losses[-1]
        barrier = max([l - (loss_a * (1 - c) + loss_b * c) for c, l in loss_dict.items()])
    
        # 3. Path Integral (AUC via Trapezoidal Rule)
        # Using np.trapezoid for NumPy 2.0+ compatibility
        auc = np.trapezoid(losses, coeffs)
    
        # 4. Endpoint Degradation (Avg loss at exactly λ=0.0 and λ=1.0)
        endpoints = (loss_a + loss_b) / 2.0
    
        return [midpoint, barrier, auc, endpoints]

    # Compile the data for all 4 conditions
    metrics_data = {
        "Gaussian 𝓝(0.5, 0.1)": calculate_theoretical_metrics(gaussian_final_losses),
        "Uniform [0.4, 0.6] (Narrow)": calculate_theoretical_metrics(narrow_final_losses),
        "Uniform [0.0, 1.0] (Broad)": calculate_theoretical_metrics(broad_final_losses),
        "Fixed (α=0.5)": calculate_theoretical_metrics(fixed_final_losses),
    }

    # Create Pandas DataFrame
    df_theoretical = pd.DataFrame.from_dict(
        metrics_data, 
        orient='index', 
        columns=[
            "Midpoint Loss ↓", 
            "Max Barrier ↓", 
            "Path Integral (AUC) ↓", 
            "Endpoint Avg Loss ↓"
        ]
    )

    # Leaving this as the last line will render a beautiful HTML table in Marimo
    df_theoretical.round(4)
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
