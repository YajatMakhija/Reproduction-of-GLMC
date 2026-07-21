"""Loss-barrier evaluation along the weight-interpolation path.

Barrier definition (paper Eq. 1):
    B_lambda = L(interp at lambda) - [ lambda*L(model0) + (1-lambda)*L(model1) ]
    B        = max_lambda B_lambda
with the interpolation grid lambda in {0.0, 0.1, ..., 1.0} (11 points), where
coeff=1 -> model0 and coeff=0 -> model1.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model, loader, device, coeff=None, max_batches=None):
    """Return (accuracy%, mean cross-entropy). coeff=None => plain model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x) if coeff is None else model(x, coeff=coeff)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return 100.0 * correct / total, loss_sum / total


def interpolate_state_dicts(sd0, sd1, coeff):
    """coeff*sd0 + (1-coeff)*sd1 for every tensor."""
    return {k: coeff * sd0[k] + (1 - coeff) * sd1[k] for k in sd0}


def summarize(coeff_losses):
    """Given {coeff: loss}, return (max_barrier, midpoint_loss, per-coeff barriers)."""
    L0 = coeff_losses[1.0]   # coeff=1.0 -> model0
    L1 = coeff_losses[0.0]   # coeff=0.0 -> model1
    barriers = {}
    max_barrier = -float("inf")
    for c, L in coeff_losses.items():
        lam = c  # weight on model0
        expected = lam * L0 + (1 - lam) * L1
        barriers[c] = L - expected
        max_barrier = max(max_barrier, barriers[c])
    return max_barrier, coeff_losses[0.5], barriers


def sweep_state_dict(model_ctor, sd0, sd1, loader, device, coeffs=None, max_batches=None):
    """VAN / WM path: interpolate two plain state dicts and evaluate."""
    if coeffs is None:
        coeffs = [round(0.1 * i, 1) for i in range(11)]
    model = model_ctor().to(device)
    losses, accs = {}, {}
    for c in coeffs:
        model.load_state_dict(interpolate_state_dicts(sd0, sd1, c))
        acc, loss = evaluate(model, loader, device, coeff=None, max_batches=max_batches)
        losses[c], accs[c] = loss, acc
    return losses, accs


def sweep_merger(merger, loader, device, coeffs=None, max_batches=None):
    """LM path: evaluate the trained ViTMerger at each coeff."""
    if coeffs is None:
        coeffs = [round(0.1 * i, 1) for i in range(11)]
    losses, accs = {}, {}
    for c in coeffs:
        acc, loss = evaluate(merger, loader, device, coeff=c, max_batches=max_batches)
        losses[c], accs[c] = loss, acc
    return losses, accs
