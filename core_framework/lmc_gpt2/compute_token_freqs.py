import argparse
import os
from itertools import chain

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory of DatasetDict saved via save_to_disk.")
    p.add_argument("--tokenizer_dir", type=str, required=True,
                   help="Tokenizer directory (e.g., ./gpt2_tokenizer).")
    p.add_argument("--out_name", type=str, default="token_freqs.pt",
                   help="Filename for the saved tensor inside splits_dir.")
    return p.parse_args()


def main():
    args = parse_args()

    ds = load_from_disk(args.splits_dir)
    train_ds = ds["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    col = train_ds["text"]
    if not isinstance(col, list):
        col = list(col)
    texts = [str(t) for t in col]

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )
    all_ids = list(chain.from_iterable(enc["input_ids"]))
    if not all_ids:
        raise ValueError("No tokens found in TRAIN split when computing frequencies.")

    ids_tensor = torch.tensor(all_ids, dtype=torch.long)
    freqs = torch.bincount(ids_tensor, minlength=tokenizer.vocab_size)

    out_path = os.path.join(args.splits_dir, args.out_name)
    torch.save(freqs, out_path)
    print(f"✅ Saved token frequencies to: {out_path}")


if __name__ == "__main__":
    main()
