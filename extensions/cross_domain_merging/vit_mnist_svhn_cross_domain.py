import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import subprocess
    import sys
    print("Installing required dependencies...")
    # This safely installs POT (which provides 'ot'), einops, and tqdm in your new environment
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'POT', 'einops', 'pyyaml', 'tqdm', '-q'])
    print("✅ Dependencies installed successfully!")
    return (sys,)


@app.cell
def _(sys):
    import os

    import copy
    import torch
    import torch.nn as nn
    import numpy as np
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # ==========================================
    # 1. QUICK SETUP & DATA LOADING
    # ==========================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = {'image_size': 32, 'patch_size': 4, 'dim': 256, 'depth': 6, 'heads': 8, 'mlp_dim': 512, 'dropout': 0.1, 'emb_dropout': 0.1, 'dim_head': 64, 'batch_size': 128}
    num_classes = 10

    if "lmc_vit" not in sys.path:
        sys.path.insert(0, os.path.abspath("lmc_vit"))

    from models import build_vit, reparameterize
    from weight_matching import weight_matching
    from merger import ViTMerger as SingleViTMerger
    from dual_merger import ViTMerger as DualViTMerger

    mnist_transform = transforms.Compose([transforms.Pad(2), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    svhn_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_dir = "lmc_vit/data"
    mnist_train_loader = DataLoader(torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_transform), batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    mnist_test_loader = DataLoader(torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=mnist_transform), batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    svhn_train_loader = DataLoader(torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=svhn_transform), batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    svhn_test_loader = DataLoader(torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=svhn_transform), batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    def evaluate_model_simple(model_fn, loader, is_lm=False, lam=None):
        criterion = nn.CrossEntropyLoss()
        loss, corr, total = 0.0, 0, 0
        eval_fn = (lambda x: model_fn(x, coeff=lam)) if is_lm else model_fn
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = eval_fn(images)
                if isinstance(out, tuple): out = out[0]
                loss += criterion(out, labels).item() * labels.size(0)
                corr += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return {"loss": loss / total, "acc": (corr / total) * 100}

    # Load Pre-trained Base Models
    model_a_base = build_vit(cfg, num_classes).to(DEVICE)
    model_b_base = build_vit(cfg, num_classes).to(DEVICE)
    model_a_base.load_state_dict(torch.load("outputs_cross_domain/mnist_150ep.pth", map_location=DEVICE))
    model_b_base.load_state_dict(torch.load("outputs_cross_domain/svhn_150ep.pth", map_location=DEVICE))
    model_a = reparameterize(model_a_base, num_classes).to(DEVICE).eval()
    model_b = reparameterize(model_b_base, num_classes).to(DEVICE).eval()

    def train_single_cross_domain_lm(model_a, model_b_wm, heads, save_path, epochs=15):
        merger = SingleViTMerger(model_a, model_b_wm, heads, device=DEVICE).to(DEVICE)
        if os.path.exists(save_path):
            print(f"Loading Single LM from {save_path}")
            merger.load_state_dict(torch.load(save_path, map_location=DEVICE))
            return merger.eval()
        
        print(f"Training Single Cross-Domain LM for {epochs} epochs...")
        optimizer = torch.optim.AdamW(merger.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epochs):
            merger.train()
            total_loss = 0
            mnist_iter, svhn_iter = iter(mnist_train_loader), iter(svhn_train_loader)
            min_batches = min(len(mnist_train_loader), len(svhn_train_loader))
            for _ in range(min_batches):
                imgs_m, lbls_m = next(mnist_iter); imgs_m, lbls_m = imgs_m.to(DEVICE), lbls_m.to(DEVICE)
                out_m = merger(imgs_m, coeff=0.5); loss_m = criterion(out_m[0] if isinstance(out_m, tuple) else out_m, lbls_m)
            
                imgs_s, lbls_s = next(svhn_iter); imgs_s, lbls_s = imgs_s.to(DEVICE), lbls_s.to(DEVICE)
                out_s = merger(imgs_s, coeff=0.5); loss_s = criterion(out_s[0] if isinstance(out_s, tuple) else out_s, lbls_s)
            
                loss = (loss_m + loss_s) / 2.0
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            print(f"  Single LM Epoch {ep+1}/{epochs} | Avg Joint Loss: {total_loss/min_batches:.4f}")
        torch.save(merger.state_dict(), save_path)
        return merger.eval()

    # ==========================================
    # 2. THE 4-WAY EVALUATION LOOP
    # ==========================================
    print("Evaluating all 4 methods (Saving Loss and Accuracy)...")
    _lambdas = np.linspace(0, 1, 11)
    results = {'vanilla': {}, 'wm': {}, 'single_lm': {}, 'dual_lm': {}}

    model_b_wm = copy.deepcopy(model_b)
    try: weight_matching(model_a, model_b_wm, cfg['heads'], iterations=15)
    except: pass

    print("--- Vanilla ---")
    for _lam in _lambdas:
        _m = copy.deepcopy(model_a); _s = _m.state_dict()
        for _k in _s: _s[_k] = _lam * model_a.state_dict()[_k] + (1 - _lam) * model_b.state_dict()[_k]
        _m.load_state_dict(_s)
        res_m = evaluate_model_simple(_m.eval(), mnist_test_loader)
        res_s = evaluate_model_simple(_m.eval(), svhn_test_loader)
        results['vanilla'][round(_lam, 1)] = {'mnist_acc': res_m['acc'], 'svhn_acc': res_s['acc'], 'mnist_loss': res_m['loss'], 'svhn_loss': res_s['loss']}

    print("--- Weight Matching ---")
    for _lam in _lambdas:
        _m = copy.deepcopy(model_a); _s = _m.state_dict()
        for _k in _s: _s[_k] = _lam * model_a.state_dict()[_k] + (1 - _lam) * model_b_wm.state_dict()[_k]
        _m.load_state_dict(_s)
        res_m = evaluate_model_simple(_m.eval(), mnist_test_loader)
        res_s = evaluate_model_simple(_m.eval(), svhn_test_loader)
        results['wm'][round(_lam, 1)] = {'mnist_acc': res_m['acc'], 'svhn_acc': res_s['acc'], 'mnist_loss': res_m['loss'], 'svhn_loss': res_s['loss']}

    print("--- Single Learned Matching ---")
    merger_single = train_single_cross_domain_lm(model_a, model_b_wm, cfg['heads'], "outputs_cross_domain/single_lm_mnist_svhn.pth")
    for _lam in _lambdas:
        res_m = evaluate_model_simple(merger_single, mnist_test_loader, is_lm=True, lam=_lam)
        res_s = evaluate_model_simple(merger_single, svhn_test_loader, is_lm=True, lam=_lam)
        results['single_lm'][round(_lam, 1)] = {'mnist_acc': res_m['acc'], 'svhn_acc': res_s['acc'], 'mnist_loss': res_m['loss'], 'svhn_loss': res_s['loss']}

    print("--- Dual Learned Matching ---")
    merger_dual = DualViTMerger(model_a, model_b_wm, cfg['heads'], device=DEVICE).to(DEVICE)
    merger_dual.load_state_dict(torch.load("outputs_cross_domain/dual_lm_mnist_svhn.pth", map_location=DEVICE))
    merger_dual.eval()
    for _lam in _lambdas:
        res_m = evaluate_model_simple(merger_dual, mnist_test_loader, is_lm=True, lam=_lam)
        res_s = evaluate_model_simple(merger_dual, svhn_test_loader, is_lm=True, lam=_lam)
        results['dual_lm'][round(_lam, 1)] = {'mnist_acc': res_m['acc'], 'svhn_acc': res_s['acc'], 'mnist_loss': res_m['loss'], 'svhn_loss': res_s['loss']}

    torch.save(results, "outputs_cross_domain/cross_domain_results.pth")
    print("✅ Done! 4-way comparison saved.")
    return np, torch


@app.cell
def _(np, torch):

    import matplotlib.pyplot as plt


    results_path = "outputs_cross_domain/cross_domain_results.pth"
    result = torch.load(results_path, map_location='cpu', weights_only=False)
    _lambdas = np.linspace(0, 1, 11)

    def get_metric(method, dataset, metric):
        return [result[method][round(lam, 1)][f'{dataset}_{metric}'] for lam in _lambdas]

    def calc_barrier(data_array, is_loss=True):
        mid = data_array[5]
        avg = (data_array[0] + data_array[-1]) / 2.0
        return (mid - avg) if is_loss else (avg - mid) # Drop for accuracy, Spike for loss

    # --- CONFIG ---
    methods = ['vanilla', 'wm', 'single_lm', 'dual_lm']
    labels = ['Vanilla', 'Weight Matching', 'Single Learned Matching', 'Dual Learned Matching']
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4']
    styles = ['--', '--', '-', '-']
    markers = ['d', 'v', 's', 'o']

    # ==========================================
    # 1. PRINT NUMERIC BARRIERS
    # ==========================================
    print("="*60)
    print("📉 LOSS BARRIERS (Lower/Negative is better)")
    print("="*60)
    for i, m in enumerate(methods):
        m_loss = get_metric(m, 'mnist', 'loss')
        s_loss = get_metric(m, 'svhn', 'loss')
        print(f"{labels[i]:25s} -> MNIST: {calc_barrier(m_loss, True):>7.4f} | SVHN: {calc_barrier(s_loss, True):>7.4f}")

    print("\n" + "="*60)
    print("🎯 ACCURACY DROPS (Lower/Negative is better)")
    print("="*60)
    for i, m in enumerate(methods):
        m_acc = get_metric(m, 'mnist', 'acc')
        s_acc = get_metric(m, 'svhn', 'acc')
        print(f"{labels[i]:25s} -> MNIST: {calc_barrier(m_acc, False):>7.2f}% | SVHN: {calc_barrier(s_acc, False):>7.2f}%")

    # ==========================================
    # 2. GENERATE PLOTS
    # ==========================================
    plt.style.use('default')

    # --- FIGURE 1: LOSS CURVES ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    for i, m in enumerate(methods):
        lw = 2.5 if 'dual' in m else 1.5
        axes1[0].plot(_lambdas, get_metric(m, 'mnist', 'loss'), color=colors[i], marker=markers[i], label=labels[i], linestyle=styles[i], linewidth=lw)
        axes1[1].plot(_lambdas, get_metric(m, 'svhn', 'loss'), color=colors[i], marker=markers[i], label=labels[i], linestyle=styles[i], linewidth=lw)

    axes1[0].set_title("Loss on MNIST (Sterile Domain)")
    axes1[0].set_xlabel(r"Interpolation ($\lambda$)")
    axes1[0].set_ylabel(r"Loss $\downarrow$")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    axes1[1].set_title("Loss on SVHN (Real-World Domain)")
    axes1[1].set_xlabel(r"Interpolation ($\lambda$)")
    axes1[1].set_ylabel(r"Loss $\downarrow$")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()

    plt.tight_layout()
    plt.savefig("outputs_cross_domain/cross_domain_loss_4way.pdf", dpi=300)
    plt.show()

    # --- FIGURE 2: ACCURACY CURVES ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for i, m in enumerate(methods):
        lw = 2.5 if 'dual' in m else 1.5
        axes2[0].plot(_lambdas, get_metric(m, 'mnist', 'acc'), color=colors[i], marker=markers[i], label=labels[i], linestyle=styles[i], linewidth=lw)
        axes2[1].plot(_lambdas, get_metric(m, 'svhn', 'acc'), color=colors[i], marker=markers[i], label=labels[i], linestyle=styles[i], linewidth=lw)

    axes2[0].set_title("Accuracy on MNIST (Sterile Domain)")
    axes2[0].set_xlabel(r"Interpolation ($\lambda$)")
    axes2[0].set_ylabel(r"Accuracy (%) $\uparrow$")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    axes2[1].set_title("Accuracy on SVHN (Real-World Domain)")
    axes2[1].set_xlabel(r"Interpolation ($\lambda$)")
    axes2[1].set_ylabel(r"Accuracy (%) $\uparrow$")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig("outputs_cross_domain/cross_domain_acc_4way.pdf", dpi=300)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
