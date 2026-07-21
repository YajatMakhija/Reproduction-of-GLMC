import argparse
import json
import os
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

import matplotlib.pyplot as plt

from merger import GPTMerger, GPTMergerWrapper

# Optional safetensors
try:
    from safetensors.torch import load_file as load_safetensors_file
    HAS_SFT = True
except Exception:
    HAS_SFT = False


# ---------- Helpers ----------

def load_state_dict_generic(model_dir: str) -> dict:
    """
    Load a state_dict from a HF checkpoint directory.
    Supports:
      - model.safetensors (preferred)
      - pytorch_model.bin
    """
    sft = os.path.join(model_dir, "model.safetensors")
    ptb = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(sft):
        if not HAS_SFT:
            raise RuntimeError("Found model.safetensors but safetensors is not installed.")
        return load_safetensors_file(sft)
    elif os.path.isfile(ptb):
        return torch.load(ptb, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_dir}")


def are_state_dicts_compatible(sd_a: dict, sd_b: dict) -> bool:
    """
    Return True if the two state dicts are shape-compatible for vanilla interpolation.
    We require that for any shared float tensor key, shapes match.
    """
    for k, v_a in sd_a.items():
        v_b = sd_b.get(k, None)
        if v_b is None:
            continue
        if torch.is_floating_point(v_a) and torch.is_floating_point(v_b):
            if v_a.shape != v_b.shape:
                print(f"[vanilla] Incompatible tensor for key '{k}': {v_a.shape} vs {v_b.shape}")
                return False
    return True


def build_datasets_from_splits(splits_dir, tokenizer, block_size):
    """
    Load precomputed raw splits and tokenize/chunk each split separately.

    Assumes `splits_dir` is a DatasetDict with "train" / "validation" / "test"
    saved via `datasets.DatasetDict.save_to_disk`.
    """
    ds = load_from_disk(splits_dir)
    raw_train, raw_val, raw_test = ds["train"], ds["validation"], ds["test"]

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=True)

    def group_texts(examples):
        ids = list(chain(*examples["input_ids"]))
        masks = list(chain(*examples["attention_mask"]))
        chunk_len = (len(ids) // block_size) * block_size
        ids = ids[:chunk_len]
        masks = masks[:chunk_len]
        return {
            "input_ids": [ids[i:i + block_size] for i in range(0, chunk_len, block_size)],
            "attention_mask": [masks[i:i + block_size] for i in range(0, chunk_len, block_size)],
        }

    def prep(one_split):
        t = one_split.map(tokenize_fn, batched=True, remove_columns=["text"])
        c = t.map(group_texts, batched=True)
        c.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return c

    train_ds = prep(raw_train)
    val_ds = prep(raw_val)
    test_ds = prep(raw_test)
    return train_ds, val_ds, test_ds


def evaluate_model(model: GPT2LMHeadModel, trainer: Trainer) -> float:
    """Run a single eval pass and return eval_loss."""
    model.eval()
    trainer.model = model
    metrics = trainer.evaluate()
    if "eval_loss" not in metrics:
        raise KeyError("Trainer returned metrics without 'eval_loss'.")
    return float(metrics["eval_loss"])


def interpolate_state_dict(sd_a: dict, sd_b: dict, lam: float) -> dict:
    """
    Vanilla interpolation between two state dicts:

        θ(λ) = λ θ_A + (1 − λ) θ_B

    Assumes keys match and shapes are compatible (checked beforehand).
    Non-floating tensors are taken from θ_A.
    """
    out = {}
    for k, v_a in sd_a.items():
        v_b = sd_b.get(k, None)
        if v_b is None:
            out[k] = v_a
            continue
        if torch.is_floating_point(v_a) and torch.is_floating_point(v_b):
            out[k] = lam * v_a + (1.0 - lam) * v_b
        else:
            out[k] = v_a
    return out


# ---------- Main eval + plot ----------

def run_eval(
    merger_wrapper: GPTMergerWrapper,
    vanilla_model: GPT2LMHeadModel,
    sd_a: dict,
    sd_b: dict,
    tokenizer,
    test_ds,
    permutations_only: bool,
    can_vanilla: bool,
    args,
):
    """
    1) Sweep with merger_wrapper BEFORE loading trained state dict  → "Weight matching"
    2) Load trained state dict into merger_wrapper and sweep again   → "Learned matching"
    3) Sweep vanilla simple interpolation model                      → "Vanilla averaging"

    Saves JSON + plot.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merger_wrapper = merger_wrapper.to(device)
    vanilla_model = vanilla_model.to(device)

    eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_drop_last=False,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=vanilla_model,  # dummy; will be overwritten
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    coeffs = []
    c = args.coeff_start
    while c <= args.coeff_end + 1e-9:
        coeffs.append(float(round(c, 10)))
        c += args.coeff_step
    coeffs = np.array(coeffs, dtype=float)

    # --- 1) Weight matching sweep (merger before loading checkpoint) ---
    print("\n=== Sweep: Weight matching (pre-trained checkpoint) ===")
    coeff_losses_weight = {}
    for coeff in coeffs:
        if hasattr(merger_wrapper, "merger_model") and hasattr(merger_wrapper.merger_model, "set_sampler"):
            merger_wrapper.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(merger_wrapper, "set_sampler"):
            merger_wrapper.set_sampler(sampler_type=None, fixed_coeff=float(coeff))

        key = f"{coeff:.6f}"
        loss = evaluate_model(merger_wrapper, trainer)
        coeff_losses_weight[key] = loss
        print(f"[Weight matching] λ={key} -> eval_loss={loss:.6f}")

    # --- 2) Load trained merger checkpoint and sweep again (learned matching) ---
    print("\n=== Loading trained merger state dict (learned matching) ===")
    sd_trained = load_state_dict_generic(args.merged_model_dir)
    missing, unexpected = merger_wrapper.load_state_dict(sd_trained, strict=False)
    if missing or unexpected:
        print(f"⚠️ Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    print("\n=== Sweep: Learned matching (post-training checkpoint) ===")
    coeff_losses_learned = {}
    for coeff in coeffs:
        if hasattr(merger_wrapper, "merger_model") and hasattr(merger_wrapper.merger_model, "set_sampler"):
            merger_wrapper.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(merger_wrapper, "set_sampler"):
            merger_wrapper.set_sampler(sampler_type=None, fixed_coeff=float(coeff))

        key = f"{coeff:.6f}"
        loss = evaluate_model(merger_wrapper, trainer)
        coeff_losses_learned[key] = loss
        print(f"[Learned matching] λ={key} -> eval_loss={loss:.6f}")

    # --- 3) Vanilla interpolation sweep (if compatible) ---
    coeff_losses_vanilla = None
    if can_vanilla:
        print("\n=== Sweep: Vanilla interpolation (θ(λ) = λ θ_A + (1 − λ) θ_B) ===")
        coeff_losses_vanilla = {}
        for coeff in coeffs:
            key = f"{coeff:.6f}"
            sd_interp = interpolate_state_dict(sd_a, sd_b, lam=float(coeff))
            vanilla_model.load_state_dict(sd_interp, strict=False)
            loss = evaluate_model(vanilla_model, trainer)
            coeff_losses_vanilla[key] = loss
            print(f"[Vanilla] λ={key} -> eval_loss={loss:.6f}")
    else:
        print("\n⚠️ Skipping Vanilla interpolation: base model state dicts are not shape-compatible.")

    # --- Save JSON payload ---
    json_path = os.path.join(args.output_dir, args.results_name)
    payload = {
        "coeffs": coeffs.tolist(),
        "coeff_losses_weight_matching": coeff_losses_weight,
        "coeff_losses_learned_matching": coeff_losses_learned,
        "coeff_start": args.coeff_start,
        "coeff_end": args.coeff_end,
        "coeff_step": args.coeff_step,
        "permutations_only": permutations_only,
        "vanilla_supported": bool(can_vanilla),
    }
    if coeff_losses_vanilla is not None:
        payload["coeff_losses_vanilla"] = coeff_losses_vanilla

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\n✅ Wrote sweep losses to: {json_path}")

    # --- Prepare arrays for plotting ---
    keys_sorted = sorted(coeff_losses_learned.keys(), key=lambda k: float(k))
    xs = np.array([float(k) for k in keys_sorted])
    ys_weight = np.array([coeff_losses_weight[k] for k in keys_sorted])
    ys_learned = np.array([coeff_losses_learned[k] for k in keys_sorted])

    if can_vanilla and coeff_losses_vanilla is not None:
        ys_vanilla = np.array([coeff_losses_vanilla[k] for k in keys_sorted])
    else:
        ys_vanilla = None

    # ---------- Loss plot ----------
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlabel("Interpolation coefficient ($\\lambda$)", fontsize=12)
    ax.set_ylabel("Loss ↓", fontsize=12)

    # Labels depending on permutations_only
    if permutations_only:
        label_weight = "Weight (perm)"
        label_learned = "Learned (perm)"
    else:
        label_weight = "Weight"
        label_learned = "Learned"

    labels = [label_weight, label_learned]
    ys_list = [ys_weight, ys_learned]

    if ys_vanilla is not None:
        labels.append("Vanilla")
        ys_list.append(ys_vanilla)

    colours = ["tab:orange", plt.cm.get_cmap("GnBu_r", 5)(1), "tab:green"]
    markers = ["v", "o", "D"]

    for i, (y, lab) in enumerate(zip(ys_list, labels)):
        ax.plot(
            xs,
            y,
            color=colours[i],
            marker=markers[i],
            markersize=5,
            markeredgecolor="black",
            linewidth=1.5,
            label=lab,
        )

    y_band_src = ys_list[1]
    if np.isfinite(y_band_src).any():
        ymin = np.nanmin(y_band_src)
        ymax = np.nanmax(y_band_src)
        band = 0.01 * (ymax - ymin) if ymax > ymin else 0.0
        if band > 0:
            ax.fill_between(xs, y_band_src - band, y_band_src + band, alpha=0.2, color=colours[1])

    xticks = np.round(np.linspace(0.0, 1.0, 6), 1)  # 0.0, 0.2, ..., 1.0
    if can_vanilla:
        xticklabels = [r"$\pi(\Theta_B)$"] + [f"{t:.1f}" for t in xticks[1:-1][::-1]] + [r"$\Theta_A$"]
    else:
        # Different-size models: use Θ↓ ... Θ↑
        xticklabels = [r"$\pi(\Theta_{\downarrow})$"] + [f"{t:.1f}" for t in xticks[1:-1][::-1]] + [r"$\Theta_{\uparrow}$"]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.invert_xaxis()

    ax.grid(True, linestyle="dotted", alpha=0.5)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        handlelength=2,
        handletextpad=0.5,
        borderpad=0.4,
    )

    fname_base = "loss_interp"

    loss_pdf = Path(args.output_dir) / f"{fname_base}.pdf"
    loss_png = Path(args.output_dir) / f"{fname_base}.png"
    fig.savefig(loss_pdf, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(loss_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Saved loss sweep plot to: {loss_pdf} and {loss_png}")

    return json_path, str(loss_png)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained GPTMerger checkpoint via coefficient sweep "
            "using merge_meta.json (weight matching vs learned matching vs optional vanilla)."
        )
    )

    # Merged checkpoint + metadata (required)
    p.add_argument(
        "--merged_model_dir",
        type=str,
        required=True,
        help="Directory of the saved merged checkpoint (Trainer.save_model output, containing merge_meta.json).",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="merger_eval_results",
        help="Where to save JSON + plots.",
    )
    p.add_argument(
        "--results_name",
        type=str,
        default="merged_coeff_losses_weight_learned_vanilla.json",
        help="Filename for JSON results (inside output_dir).",
    )

    # Sweep / eval knobs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--coeff_start", type=float, default=0.0)
    p.add_argument("--coeff_end", type=float, default=1.0)
    p.add_argument("--coeff_step", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Load metadata from merged_model_dir ---
    meta_path = os.path.join(args.merged_model_dir, "merge_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Expected metadata at {meta_path} but did not find it.\n"
            "Make sure your training script saves merge_meta.json into output_dir."
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_dir_0 = meta["model_dir_0"]
    model_dir_1 = meta["model_dir_1"]
    permutations_only = bool(meta.get("permutations_only", False))
    token_freqs_path = meta.get("token_freqs_path", None)

    tokenizer_dir = meta.get("tokenizer_dir", None)
    splits_dir = meta.get("splits_dir", None)
    if tokenizer_dir is None:
        raise ValueError("tokenizer_dir missing in merge_meta.json.")
    if splits_dir is None:
        raise ValueError("splits_dir missing in merge_meta.json.")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    _, _, test_ds = build_datasets_from_splits(
        splits_dir=splits_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    # --- Base state dicts for vanilla / compatibility check ---
    sd_a = load_state_dict_generic(model_dir_0)  # θ_A
    sd_b = load_state_dict_generic(model_dir_1)  # θ_B

    can_vanilla = are_state_dicts_compatible(sd_a, sd_b)

    # Vanilla model
    vanilla_model = GPT2LMHeadModel.from_pretrained(model_dir_0)
    vanilla_model.eval()
    try:
        vanilla_model.config.attn_implementation = "eager"
        vanilla_model._attn_implementation = "eager"
    except Exception:
        pass

    # --- Base models for merger wrapper---
    base0_cpu = GPT2LMHeadModel.from_pretrained(model_dir_0)
    base1_cpu = GPT2LMHeadModel.from_pretrained(model_dir_1)
    base0_cpu.eval()
    base1_cpu.eval()

    for m in (base0_cpu, base1_cpu):
        try:
            m.config.attn_implementation = "eager"
            m._attn_implementation = "eager"
        except Exception:
            pass

    token_freqs = None
    if token_freqs_path and os.path.isfile(token_freqs_path):
        token_freqs = torch.load(token_freqs_path, map_location="cpu")

    merger_model = GPTMerger(
        base0_cpu,
        base1_cpu,
        token_freqs=token_freqs,
        permutations_only=permutations_only,
    )
    merger_wrapper = GPTMergerWrapper(config=base0_cpu.config, merger_model=merger_model)

    run_eval(
        merger_wrapper=merger_wrapper,
        vanilla_model=vanilla_model,
        sd_a=sd_a,
        sd_b=sd_b,
        tokenizer=tokenizer,
        test_ds=test_ds,
        permutations_only=permutations_only,
        can_vanilla=can_vanilla,
        args=args,
    )

if __name__ == "__main__":
    main()
