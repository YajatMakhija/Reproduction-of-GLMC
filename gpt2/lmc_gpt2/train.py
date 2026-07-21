from itertools import chain
import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorForLanguageModeling,
)

import wandb


def parse_args():
    p = argparse.ArgumentParser()
    # REQUIRED: load precomputed splits saved via save_to_disk (must contain train/validation/test)
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory of a saved DatasetDict with train/validation/test (from save_to_disk).")
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer",
                   help="Directory of a fixed GPT-2 tokenizer (same across seeds).")
    p.add_argument("--seed", type=int, default=1, help="Random seed for this run.")
    p.add_argument("--block_size", type=int, default=256, help="Context length.")
    p.add_argument("--n_layer", type=int, default=6, help="Transformer layers.")
    p.add_argument("--n_embd", type=int, default=256, help="Embedding/hidden size.")
    p.add_argument("--n_inner", type=int, default=1024, help="Inner dimension of the FFN.")
    p.add_argument("--n_head", type=int, default=4, help="Attention heads (must divide n_embd).")
    p.add_argument("--tie_word_embeddings", action="store_true", help="Tie word embeddings.")
    p.add_argument("--batch_size", type=int, default=32, help="Per-device train batch size.")
    p.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    p.add_argument("--eval_steps", type=int, default=50, help="Eval every N steps.")
    p.add_argument("--logging_steps", type=int, default=25, help="Log every N steps.")
    p.add_argument("--save_total_limit", type=int, default=2, help="Limit saved checkpoints.")
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping (on validation).")
    p.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (in evals).")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if CUDA is available.")
    p.add_argument("--output_dir", type=str, default=None, help="Where to save outputs.")
    # W&B
    p.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="gpt2-merging-demo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="merge")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_dir or f"gpt2_tinyshakespeare_seed{args.seed}_nembd{args.n_embd}"
    os.makedirs(out_dir, exist_ok=True)

    # --- init W&B (optional) ---
    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="train",
            config=vars(args),  # all CLI args
            tags=[t for t in args.wandb_tags.split(",") if t],
            name=f"gpt2-train-seed{args.seed}"
        )

    # 1) Load precomputed splits (train/validation/test)
    ds = load_from_disk(args.splits_dir)
    raw_train, raw_val, raw_test = ds["train"], ds["validation"], ds["test"]

    # 2) Fixed tokenizer across runs
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token  # ensure padding token exists for collator
    block_size = args.block_size

    # 3) Tokenization & grouping (done per split)
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

    def prep(ds_split):
        t = ds_split.map(tokenize_fn, batched=True, remove_columns=["text"])
        c = t.map(group_texts, batched=True)
        c.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return c

    chunked_train = prep(raw_train)
    chunked_val   = prep(raw_val)
    chunked_test  = prep(raw_test)

    # 4) Collator (causal LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Model config
    configuration = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        tie_word_embeddings=args.tie_word_embeddings,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(configuration)

    # 6) TrainingArguments: validate/early-stop on VALIDATION
    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,                 # save whenever we evaluate
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,                # keep best eval_loss (on validation)
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
        report_to=("wandb" if args.wandb else "none"),
        run_name=f"gpt2-tinyshakespeare-seed{args.seed}-n_embd{args.n_embd}",
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=chunked_train,
        eval_dataset=chunked_val,  
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 7) Train
    trainer.train()

    trainer.save_model(out_dir)         
    tokenizer.save_pretrained(out_dir)

    # Record best checkpoint path
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is not None:
        with open(os.path.join(out_dir, "BEST_CHECKPOINT.txt"), "w") as f:
            f.write(best_ckpt + "\n")

    # 8) One-shot final evaluation on TEST (held out)
    test_metrics = trainer.evaluate(eval_dataset=chunked_test, metric_key_prefix="test")
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("TEST metrics:", test_metrics)

    # --- W&B: record final metrics & log artifact ---
    if run is not None:
        art = wandb.Artifact(
            name=f"gpt2-model-seed{args.seed}",
            type="model",
            metadata={
                "seed": args.seed,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "splits_dir": args.splits_dir,
                "tokenizer_dir": args.tokenizer_dir,
                "best_checkpoint": best_ckpt,
            },
        )
        art.add_dir(out_dir)
        aliases = ["latest", f"seed-{args.seed}"]
        if best_ckpt: aliases.append("best")
        run.log_artifact(art, aliases=aliases)
        run.finish()


if __name__ == "__main__":
    main()