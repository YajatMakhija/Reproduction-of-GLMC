import argparse
import json
import os
from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", type=str, required=True,
                   help="Path to a plain-text corpus (e.g., tinyshakespeare.txt).")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to save the DatasetDict (will be created).")
    p.add_argument("--val_frac", type=float, default=0.025,
                   help="Validation fraction (0..1).")
    p.add_argument("--test_frac", type=float, default=0.025,
                   help="Test fraction (0..1).")
    p.add_argument("--method", type=str, default="contiguous",
                   choices=["contiguous", "random"],
                   help="Splitting method. 'contiguous' splits by character offsets; "
                        "'random' splits by examples (lines).")
    p.add_argument("--split_seed", type=int, default=123,
                   help="Seed for random splitting (only used with --method random).")
    return p.parse_args()


def contiguous_splits_from_text(text: str, val_frac: float, test_frac: float) -> Tuple[str, str, str]:
    n = len(text)
    if n == 0:
        raise ValueError("Input text file is empty.")

    if not (0 <= val_frac < 1) or not (0 <= test_frac < 1) or (val_frac + test_frac >= 1):
        raise ValueError("val_frac and test_frac must be in [0,1) and sum to < 1.")

    test_n = int(round(n * test_frac))
    val_n = int(round(n * val_frac))
    # Ensure boundaries are valid
    holdout_n = val_n + test_n
    if holdout_n >= n:
        raise ValueError("Holdout size is >= total text length; reduce val/test fractions.")

    train_text = text[: n - holdout_n]
    val_text = text[n - holdout_n: n - test_n] if val_n > 0 else ""
    test_text = text[n - test_n:] if test_n > 0 else ""

    return train_text, val_text, test_text


def build_contiguous_datasetdict(data_file: str, val_frac: float, test_frac: float) -> DatasetDict:
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    train_text, val_text, test_text = contiguous_splits_from_text(text, val_frac, test_frac)

    def to_ds(s: str) -> Dataset:
        return Dataset.from_dict({"text": [s]})

    ds = DatasetDict({
        "train": to_ds(train_text),
        "validation": to_ds(val_text),
        "test": to_ds(test_text),
    })
    return ds


def build_random_datasetdict(data_file: str, val_frac: float, test_frac: float, seed: int) -> DatasetDict:
    if not (0 <= val_frac < 1) or not (0 <= test_frac < 1) or (val_frac + test_frac >= 1):
        raise ValueError("val_frac and test_frac must be in [0,1) and sum to < 1.")
    raw = load_dataset("text", data_files=data_file)["train"]

    holdout_frac = val_frac + test_frac
    tmp = raw.train_test_split(test_size=holdout_frac, seed=seed)
    train = tmp["train"]
    holdout = tmp["test"]

    inner_test_size = test_frac / holdout_frac if holdout_frac > 0 else 0.0
    inner = holdout.train_test_split(test_size=inner_test_size, seed=seed)
    val, test = inner["train"], inner["test"]

    ds = DatasetDict({"train": train, "validation": val, "test": test})
    return ds


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == "contiguous":
        ds = build_contiguous_datasetdict(args.data_file, args.val_frac, args.test_frac)
        meta = {
            "method": "contiguous",
            "data_file": os.path.abspath(args.data_file),
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "train_chars": len(ds["train"]["text"][0]),
            "val_chars": len(ds["validation"]["text"][0]) if len(ds["validation"]) else 0,
            "test_chars": len(ds["test"]["text"][0]) if len(ds["test"]) else 0,
        }
    else:
        ds = build_random_datasetdict(args.data_file, args.val_frac, args.test_frac, args.split_seed)
        meta = {
            "method": "random",
            "data_file": os.path.abspath(args.data_file),
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "split_seed": args.split_seed,
            "train_rows": len(ds["train"]),
            "val_rows": len(ds["validation"]),
            "test_rows": len(ds["test"]),
        }

    ds.save_to_disk(args.out_dir)

    meta_path = os.path.join(args.out_dir, "split_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved DatasetDict to: {args.out_dir}")
    print(f"   Splits: {list(ds.keys())}")
    print(f"   Metadata: {meta_path}")


if __name__ == "__main__":
    main()