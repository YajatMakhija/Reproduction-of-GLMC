import argparse
import json
import os
from itertools import chain

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed,
)
from enums import SamplerType

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from merger import GPTMerger, GPTMergerWrapper
import wandb 


def parse_args():
    p = argparse.ArgumentParser(
        description="Train GPTMerger on precomputed splits and evaluate merged model with a coefficient sweep."
    )
    # REQUIRED: paths for base models to merge and the saved splits
    p.add_argument("--model_dir_0", type=str, required=True)
    p.add_argument("--model_dir_1", type=str, required=True)
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory of DatasetDict (train/validation/test) saved via save_to_disk.")

    # Tokenizer / IO
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer")
    p.add_argument("--output_dir", type=str, default="gpt2_lmc_tiny_merge")
    p.add_argument("--results_name", type=str, default="merged_sampler_losses.json")

    # Train / eval knobs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default=SamplerType.NARROW_UNI)
    p.add_argument("--permutations_only", action="store_true", help="Enable only using permutations (no orthogonal alignment).")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--fp16", action="store_true")

    # Early stopping (on validation)
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping on validation.")
    p.add_argument("--early_stop_patience", type=int, default=10, help="Patience in number of evals.")

    # Sweep / eval
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--coeff_start", type=float, default=0.0)
    p.add_argument("--coeff_end", type=float, default=1.0)
    p.add_argument("--coeff_step", type=float, default=0.1)

    # Optional (but strongly recommended) token frequency prior for merger
    p.add_argument("--token_freqs_path", type=str, default="tinyshakespeare_token_freqs.pt")

    # --- W&B ---
    p.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="gpt2-merging-demo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="merge")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


def _is_artifact_ref(s: str) -> bool:
    return (":" in s) and (not os.path.isdir(s))


def _ensure_local_model_dir(run: "wandb.sdk.wandb_run.Run", spec_or_dir: str) -> str:
    """
    If `spec_or_dir` is a W&B artifact ref, download and return a local dir.
    If it's a local dir and W&B is on, log it as an input artifact for lineage.
    """
    if run is None:
        return spec_or_dir
    if _is_artifact_ref(spec_or_dir):
        art = run.use_artifact(spec_or_dir, type="model")
        return art.download()
    # local dir → upload once so the merge run has parents in the DAG
    name = f"input-{os.path.basename(os.path.abspath(spec_or_dir))}-{run.id}"
    art = wandb.Artifact(name=name, type="model",
                         metadata={"source": "local_path", "path": os.path.abspath(spec_or_dir)})
    art.add_dir(spec_or_dir)
    logged = run.log_artifact(art, aliases=["input"])  # optional; just records it
    return spec_or_dir


def build_merger_wrapper(model_dir_0, model_dir_1, sampler_type, permutations_only, token_freqs_path=None):
    model0 = GPT2LMHeadModel.from_pretrained(model_dir_0)
    model1 = GPT2LMHeadModel.from_pretrained(model_dir_1)
    model0.eval(); model1.eval()

    # Make sure attention impl is consistent
    for m in (model0, model1):
        try:
            m.config.attn_implementation = "eager"
            m._attn_implementation = "eager"
        except Exception:
            pass

    token_freqs = None
    if token_freqs_path and os.path.isfile(token_freqs_path):
        token_freqs = torch.load(token_freqs_path, map_location="cpu")

    merger_model = GPTMerger(model0, model1, token_freqs=token_freqs, permutations_only=permutations_only)
    merger_model.set_sampler(sampler_type=sampler_type)
    return GPTMergerWrapper(config=model0.config, merger_model=merger_model)


def build_datasets_from_splits(splits_dir, tokenizer, block_size):
    """Load precomputed raw splits and tokenize/chunk each split separately."""
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

    return prep(raw_train), prep(raw_val), prep(raw_test)


def run_training_and_save(args, tokenizer, train_ds, val_ds):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        report_to=("wandb" if args.wandb else "none"),  # ← minimal change
    )

    model = build_merger_wrapper(args.model_dir_0, args.model_dir_1, args.sampler, token_freqs_path=args.token_freqs_path, permutations_only=args.permutations_only)

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # VALIDATION for early stopping & best checkpoint
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    best_ckpt = trainer.state.best_model_checkpoint or args.output_dir
    with open(os.path.join(args.output_dir, "BEST_CHECKPOINT.txt"), "w") as f:
        f.write(best_ckpt + "\n")

    return trainer.model


def evaluate_sweep(model, tokenizer, test_ds, args, run: "wandb.sdk.wandb_run.Run" = None):
    """
    Run a fixed-coefficient sweep on the held-out test set and compute the empirical
    loss barrier as in Eq. (barrier):

        B[θ_A, θ_B](D) = sup_{λ in [0,1]} B_λ[θ_A, θ_B](D),

    where

        B_λ = L(λ θ_A + (1-λ) θ_B) - (λ L(θ_A) + (1-λ) L(θ_B)).

    We approximate the sup over λ using a discrete grid of coefficients.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_drop_last=False,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_ds,  # TEST ONLY
        data_collator=data_collator,
    )

    # Build coefficient grid (inclusive of end)
    coeffs = []
    c = args.coeff_start
    while c <= args.coeff_end + 1e-9:
        coeffs.append(float(round(c, 10)))  # keep numeric; avoid FP drift
        c += args.coeff_step

    # 1) Evaluate loss at each coefficient
    coeff_losses = {}
    for coeff in coeffs:
        if hasattr(model, "merger_model") and hasattr(model.merger_model, "set_sampler"):
            model.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(model, "set_sampler"):
            model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))

        metrics = eval_trainer.evaluate()
        loss = float(metrics.get("eval_loss", float("nan")))
        coeff_key = f"{coeff:.6f}"
        coeff_losses[coeff_key] = loss
        print(f"coeff={coeff_key} -> test_loss={loss:.6f}")

    # 2) Compute empirical loss barrier on this grid
    if args.coeff_end == args.coeff_start:
        raise ValueError("coeff_start and coeff_end must be different to define a barrier.")

    start_key = f"{float(args.coeff_start):.6f}"
    end_key   = f"{float(args.coeff_end):.6f}"

    if start_key not in coeff_losses or end_key not in coeff_losses:
        raise RuntimeError(
            f"Endpoint losses missing: start={start_key in coeff_losses}, end={end_key in coeff_losses}"
        )

    L_start = coeff_losses[start_key]  
    L_end   = coeff_losses[end_key] 

    coeff_barriers = {}
    max_barrier = float("-inf")
    coeff_at_max = None

    span = args.coeff_end - args.coeff_start

    for coeff in coeffs:
        key = f"{coeff:.6f}"
        L_lambda = coeff_losses[key]

        lam = (coeff - args.coeff_start) / span 
        expected_interp_loss = lam * L_end + (1.0 - lam) * L_start

        B_lambda = L_lambda - expected_interp_loss
        coeff_barriers[key] = B_lambda

        if B_lambda > max_barrier:
            max_barrier = B_lambda
            coeff_at_max = coeff

    print("\n=== Empirical loss barrier on test set ===")
    print(f"Max barrier B = {max_barrier:.6f} at coeff = {coeff_at_max:.6f}")

    # 3) Save everything to disk
    out_path = os.path.join(args.output_dir, args.results_name)
    payload = {
        "coeff_losses": coeff_losses,
        "coeff_barriers": coeff_barriers,
        "max_barrier": max_barrier,
        "coeff_at_max": coeff_at_max,
        "coeff_start": args.coeff_start,
        "coeff_end": args.coeff_end,
        "coeff_step": args.coeff_step,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"✅ Wrote sweep TEST losses and barriers to: {out_path}")

    # 4) Minimal W&B logging: just the barrier (no plots)
    if run is not None:
        run.log(
            {
                "merge/max_barrier": max_barrier,
                "merge/max_barrier_coeff": coeff_at_max,
            }
        )

    return out_path




def main():
    args = parse_args()

    meta = {
        "model_dir_0": args.model_dir_0,
        "model_dir_1": args.model_dir_1,
        "permutations_only": args.permutations_only,
        "token_freqs_path": args.token_freqs_path,
        "sampler": args.sampler,
        "tokenizer_dir": args.tokenizer_dir,
        "splits_dir": args.splits_dir,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "merge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


    set_seed(args.seed)

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="merge",
            config=vars(args),
            tags=[t for t in args.wandb_tags.split(",") if t],
            name=args.wandb_run_name,
        )
        args.model_dir_0 = _ensure_local_model_dir(run, args.model_dir_0)
        args.model_dir_1 = _ensure_local_model_dir(run, args.model_dir_1)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load splits and prep datasets
    train_ds, val_ds, test_ds = build_datasets_from_splits(
        splits_dir=args.splits_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    # Train on train, early-stop/select on validation
    trained_model = run_training_and_save(args, tokenizer, train_ds, val_ds)

    # Final sweep strictly on held-out test
    results_path = evaluate_sweep(trained_model, tokenizer, test_ds, args, run)

    # Log merged model artifact (with parents) if W&B is enabled
    if run is not None:
        merged_art = wandb.Artifact(
            name=f"gpt2-merged-{run.id}",
            type="model",
            metadata={
                "parents": {
                    "model_dir_0": os.path.abspath(args.model_dir_0),
                    "model_dir_1": os.path.abspath(args.model_dir_1),
                },
                "results_file": os.path.abspath(results_path),
            },
        )
        merged_art.add_dir(args.output_dir)
        run.log_artifact(merged_art, aliases=["latest"])
        run.finish()


if __name__ == "__main__":
    main()
