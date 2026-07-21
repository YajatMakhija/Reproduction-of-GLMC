

import os
import sys
import subprocess

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.50.1", "datasets>=2.14,<2.20", "POT", "einops",
    "safetensors", "accelerate>=0.26.0", "scipy", "matplotlib", "pandas",
    "huggingface_hub>=0.23.0",
], check=False)

import json
import math
import copy
import random
import shutil
import time
import gc
import glob
import inspect
import re
import traceback
from itertools import chain
from pathlib import Path
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange
from scipy.optimize import linear_sum_assignment
import ot

from transformers import (
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required for training/matching.")
print(
    "GPU:", torch.cuda.get_device_name(0),
    "| cap:", torch.cuda.get_device_capability(0),
    "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.cuda.set_device(0)


class CFG:
    SMOKE = False

    LOW_RAM = True
    NUM_PROC = 1
    DATA_PACK_BATCH = 400
    DATA_TOK_BATCH = 32
    DATA_CHUNK_BATCH = 16
    DATA_FREQ_BATCH = 512
    DATA_TRAIN_SHARD = 500_000

    DATASET = "rojagtap/bookcorpus"
    VAL_FRAC = 0.01
    TEST_FRAC = 0.01
    SENTS_PER_DOC = 40
    BLOCK_SIZE = 512

    N_LAYER = 6
    N_HEAD = 8
    N_EMBD = 512
    N_INNER = 2048

    TRAIN_EPOCHS = 5
    TRAIN_BS = 64
    GRAD_ACCUM = 1
    LR = 2.5e-4
    WARMUP_RATIO = 0.05
    WEIGHT_DECAY = 0.01
    MAX_STEPS = None

    MERGE_EPOCHS = 0.5
    MERGE_BS = 128
    MERGE_GRAD_ACCUM = 1
    MERGE_MAX_STEPS = None

    COEFF_START = 0.0
    COEFF_END = 1.0
    COEFF_STEP = 0.1
    EVAL_BS = 64

    BF16 = False
    FP16 = True
    SEEDS = (0, 1)
    TOKENIZER = "gpt2"
    OUT = "./glmc_bookcorpus_out"
    WEIGHT_MATCH_ITERS = 15
    MATCH_TOPK_VOCAB = None
    MATCH_ON_GPU = True
    REQUIRE_CUDA = True
    SAVE_STEPS = 200
    SAVE_TOTAL_LIMIT = 5

    SMOKE_TRAIN_BLOCKS = 64
    SMOKE_VAL_BLOCKS = 16
    SMOKE_TEST_BLOCKS = 16


cfg = CFG()
cfg.LOW_RAM = True
cfg.NUM_PROC = 1

if cfg.BF16 and torch.cuda.is_bf16_supported():
    cfg.FP16 = False
else:
    cfg.BF16 = False
    cfg.FP16 = True

if cfg.SMOKE:
    cfg.OUT = "./glmc_bookcorpus_smoke_out"
    cfg.TRAIN_EPOCHS = 1
    cfg.MERGE_EPOCHS = 1
    cfg.TRAIN_BS = 4
    cfg.MERGE_BS = 4
    cfg.EVAL_BS = 4
    cfg.GRAD_ACCUM = 1
    cfg.MERGE_GRAD_ACCUM = 1
    cfg.MAX_STEPS = 6
    cfg.MERGE_MAX_STEPS = 4
    cfg.WEIGHT_MATCH_ITERS = 2
    cfg.MATCH_TOPK_VOCAB = 2000
    cfg.COEFF_STEP = 0.5
    cfg.SAVE_STEPS = 3
    cfg.NUM_PROC = 1
    cfg.SENTS_PER_DOC = 8
    print(f"*** SMOKE MODE *** out={cfg.OUT}")
else:
    print(
        f"PAPER-LIKE | BookCorpus | Block={cfg.BLOCK_SIZE} | embd={cfg.N_EMBD} | "
        f"merge_epochs={cfg.MERGE_EPOCHS} | merge_bs={cfg.MERGE_BS} | "
        f"fp16={cfg.FP16} bf16={cfg.BF16} | out={cfg.OUT}"
    )


def _reclaim(msg=""):
    gc.collect()
    if msg:
        print(f"  [ram] {msg}", flush=True)


cache_tag = "smoke_v1" if cfg.SMOKE else "paper_v2_full"
cache_root = os.path.join(cfg.OUT, f"data_cache_{cache_tag}")
chunked_dir = os.path.join(cache_root, "chunked")
freqs_path = os.path.join(cache_root, "token_freqs.pt")
freqs_progress_path = freqs_path + ".progress.pt"
os.makedirs(cache_root, exist_ok=True)

tokenizer = None
for d in [
    os.path.join(cfg.OUT, f"gpt2_bookcorpus_seed0_nembd{cfg.N_EMBD}"),
    os.path.join(cfg.OUT, f"gpt2_bookcorpus_seed1_nembd{cfg.N_EMBD}"),
]:
    if os.path.isfile(os.path.join(d, "vocab.json")) or os.path.isfile(os.path.join(d, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(d)
        break
if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = int(10 ** 9)

nproc = 1
pack_bs = int(getattr(cfg, "DATA_PACK_BATCH", 400))
tok_bs = int(getattr(cfg, "DATA_TOK_BATCH", 32))
chunk_bs = int(getattr(cfg, "DATA_CHUNK_BATCH", 16))
freq_bs = 64
train_shard = int(getattr(cfg, "DATA_TRAIN_SHARD", 500_000))
MERGE_BATCH = 4


def _load_raw():
    last = None
    for attempt in range(1, 8):
        try:
            try:
                return load_dataset(cfg.DATASET, split="train", keep_in_memory=False)
            except TypeError:
                return load_dataset(cfg.DATASET, split="train")
        except Exception as e:
            last = e
            wait = min(120, 10 * attempt)
            print(f"  attempt {attempt}/7 failed: {type(e).__name__}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"BookCorpus download failed. Last: {last}")


def _pack_tokenize_chunk(sentence_ds, desc):
    def group_docs(examples):
        texts = examples["text"]
        docs, spd = [], cfg.SENTS_PER_DOC
        for i in range(0, len(texts), spd):
            chunk = [(t or "").strip() for t in texts[i:i + spd]]
            chunk = [t for t in chunk if t]
            if chunk:
                docs.append(" ".join(chunk))
        return {"text": docs}

    def tok(examples):
        return tokenizer(examples["text"], return_attention_mask=True, add_special_tokens=False)

    def group(examples):
        ids = list(chain(*examples["input_ids"]))
        masks = list(chain(*examples["attention_mask"]))
        nblk = (len(ids) // cfg.BLOCK_SIZE) * cfg.BLOCK_SIZE
        ids, masks = ids[:nblk], masks[:nblk]
        return {
            "input_ids": [ids[i:i + cfg.BLOCK_SIZE] for i in range(0, nblk, cfg.BLOCK_SIZE)],
            "attention_mask": [masks[i:i + cfg.BLOCK_SIZE] for i in range(0, nblk, cfg.BLOCK_SIZE)],
        }

    def _safe_map(ds, fn, **kw):
        for drop in ((), ("keep_in_memory",), ("keep_in_memory", "writer_batch_size")):
            try:
                call = dict(kw)
                for k in drop:
                    call.pop(k, None)
                return ds.map(fn, **call)
            except TypeError:
                continue
        return ds.map(fn, **{k: v for k, v in kw.items()
                             if k in ("batched", "batch_size", "remove_columns", "desc", "num_proc", "load_from_cache_file")})

    map_kw = dict(num_proc=nproc, load_from_cache_file=True, keep_in_memory=False,
                  writer_batch_size=max(100, chunk_bs))
    t0 = time.time()
    packed = _safe_map(sentence_ds, group_docs, batched=True, batch_size=pack_bs,
                        remove_columns=sentence_ds.column_names, desc=f"pack_{desc}", **map_kw)
    del sentence_ds
    _reclaim(f"after pack_{desc}")
    tokenized = _safe_map(packed, tok, batched=True, batch_size=tok_bs,
                          remove_columns=["text"], desc=f"tok_{desc}", **map_kw)
    del packed
    _reclaim(f"after tok_{desc}")
    chunked = _safe_map(tokenized, group, batched=True, batch_size=chunk_bs, desc=f"chunk_{desc}", **map_kw)
    del tokenized
    _reclaim(f"after chunk_{desc}")
    print(f"  {desc}: {len(chunked):,} blocks in {time.time() - t0:.1f}s", flush=True)
    return chunked


def count_freqs_from_shards(shard_root, vocab_size):
    shard_paths = sorted(
        os.path.join(shard_root, d)
        for d in os.listdir(shard_root)
        if d.startswith("shard_")
        and os.path.isdir(os.path.join(shard_root, d))
        and os.path.isfile(os.path.join(shard_root, d, "dataset_info.json"))
    )
    assert shard_paths, f"no complete shards in {shard_root}"

    start_si = 0
    freqs = np.zeros(vocab_size, dtype=np.int64)
    if os.path.isfile(freqs_progress_path):
        prog = torch.load(freqs_progress_path, map_location="cpu", weights_only=True)
        freqs = prog["freqs"].cpu().numpy() if torch.is_tensor(prog["freqs"]) else np.asarray(prog["freqs"])
        start_si = int(prog.get("next_shard", 0))
        print(f"  RESUME token_freqs from shard {start_si}/{len(shard_paths)}", flush=True)

    t0 = time.time()
    for si in range(start_si, len(shard_paths)):
        sp = shard_paths[si]
        ds = load_from_disk(sp)
        n = len(ds)
        for i in range(0, n, freq_bs):
            flat = np.asarray(ds[i:i + freq_bs]["input_ids"], dtype=np.int32).ravel()
            freqs += np.bincount(flat, minlength=vocab_size).astype(np.int64)
            del flat
        del ds
        _reclaim()
        torch.save({"freqs": torch.from_numpy(freqs.copy()), "next_shard": si + 1}, freqs_progress_path)

    out = torch.from_numpy(freqs)
    torch.save(out, freqs_path)
    if os.path.isfile(freqs_progress_path):
        os.remove(freqs_progress_path)
    print(f"  token_freqs done in {time.time() - t0:.1f}s -> {freqs_path}", flush=True)
    return out


def count_token_freqs_resumable(chunked, vocab_size):
    start = 0
    freqs = np.zeros(vocab_size, dtype=np.int64)
    if os.path.isfile(freqs_progress_path):
        prog = torch.load(freqs_progress_path, map_location="cpu", weights_only=True)
        freqs = prog["freqs"].cpu().numpy() if torch.is_tensor(prog["freqs"]) else np.asarray(prog["freqs"])
        start = int(prog.get("next_i", 0))

    n = len(chunked)
    t0 = time.time()
    steps = 0
    for i in range(start, n, freq_bs):
        j = min(i + freq_bs, n)
        flat = np.asarray(chunked[i:j]["input_ids"], dtype=np.int32).ravel()
        freqs += np.bincount(flat, minlength=vocab_size).astype(np.int64)
        del flat
        steps += 1
        if steps % 50 == 0 or j >= n:
            torch.save({"freqs": torch.from_numpy(freqs.copy()), "next_i": j}, freqs_progress_path)
        if steps % 200 == 0:
            _reclaim()

    out = torch.from_numpy(freqs)
    torch.save(out, freqs_path)
    if os.path.isfile(freqs_progress_path):
        os.remove(freqs_progress_path)
    print(f"  token_freqs done in {time.time() - t0:.1f}s -> {freqs_path}", flush=True)
    return out


def _merge_shard_paths(shard_paths, out_path, desc="train"):
    if os.path.isdir(out_path) and os.path.isfile(os.path.join(out_path, "dataset_info.json")):
        return load_from_disk(out_path)

    mid_root = out_path + "_merge_mid"
    os.makedirs(mid_root, exist_ok=True)
    mid_paths = []

    for bi, start in enumerate(range(0, len(shard_paths), MERGE_BATCH)):
        batch_paths = shard_paths[start:start + MERGE_BATCH]
        mid = os.path.join(mid_root, f"mid_{bi:04d}")
        if os.path.isdir(mid) and os.path.isfile(os.path.join(mid, "dataset_info.json")):
            mid_paths.append(mid)
            continue
        parts = [load_from_disk(p) for p in batch_paths]
        merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        del parts
        _reclaim(f"concat mid {bi}")
        merged.save_to_disk(mid)
        del merged
        _reclaim(f"saved mid {bi}")
        mid_paths.append(mid)

    cur = mid_paths
    round_i = 0
    while len(cur) > 1:
        nxt = []
        for bi, start in enumerate(range(0, len(cur), MERGE_BATCH)):
            batch = cur[start:start + MERGE_BATCH]
            out_mid = os.path.join(mid_root, f"r{round_i}_{bi:04d}")
            if os.path.isdir(out_mid) and os.path.isfile(os.path.join(out_mid, "dataset_info.json")):
                nxt.append(out_mid)
                continue
            parts = [load_from_disk(p) for p in batch]
            merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            del parts
            _reclaim()
            merged.save_to_disk(out_mid)
            del merged
            _reclaim()
            nxt.append(out_mid)
        cur = nxt
        round_i += 1

    if os.path.isdir(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    shutil.copytree(cur[0], out_path)
    _reclaim(f"saved {desc}")
    return load_from_disk(out_path)


def _materialize_split(sentence_ds, split_name, out_path, sharded=False):
    if os.path.isdir(out_path) and os.path.isfile(os.path.join(out_path, "dataset_info.json")):
        return load_from_disk(out_path)

    if sharded and len(sentence_ds) > train_shard:
        shard_root = out_path + "_shards"
        os.makedirs(shard_root, exist_ok=True)
        shard_paths = []
        n = len(sentence_ds)
        n_shards = (n + train_shard - 1) // train_shard
        for si, start in enumerate(range(0, n, train_shard)):
            end = min(start + train_shard, n)
            sp = os.path.join(shard_root, f"shard_{si:04d}")
            shard_paths.append(sp)
            if os.path.isdir(sp) and os.path.isfile(os.path.join(sp, "dataset_info.json")):
                continue
            piece = sentence_ds.select(range(start, end))
            chunked_piece = _pack_tokenize_chunk(piece, f"{split_name}_s{si}")
            del piece
            chunked_piece.save_to_disk(sp)
            del chunked_piece
            _reclaim(f"saved {split_name} shard {si}")
        return _merge_shard_paths(shard_paths, out_path, desc=split_name)

    chunked = _pack_tokenize_chunk(sentence_ds, split_name)
    chunked.save_to_disk(out_path)
    del chunked
    _reclaim(f"saved {split_name}")
    return load_from_disk(out_path)


def _write_dataset_dict_json(root):
    with open(os.path.join(root, "dataset_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": ["train", "validation", "test"]}, f)


def _smoke_sentences(n_sents):
    try:
        it = load_dataset(cfg.DATASET, split="train", streaming=True)
        sents = []
        for ex in it:
            t = (ex.get("text") or "").strip()
            if t:
                sents.append(t)
            if len(sents) >= n_sents:
                break
        if len(sents) >= max(32, n_sents // 4):
            return sents[:n_sents]
    except Exception:
        pass
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Once upon a time in a distant land there lived a wise king. "
        "Machine learning models learn patterns from data. "
    )
    return [(base * 20) + f" sentence {i}." for i in range(n_sents)]


train_path = os.path.join(chunked_dir, "train")
val_path = os.path.join(chunked_dir, "validation")
test_path = os.path.join(chunked_dir, "test")
shard_root = train_path + "_shards"

if (
    os.path.isdir(train_path) and os.path.isfile(os.path.join(train_path, "dataset_info.json"))
    and os.path.isdir(val_path) and os.path.isdir(test_path)
):
    chunked_train = load_from_disk(train_path)
    chunked_val = load_from_disk(val_path)
    chunked_test = load_from_disk(test_path)

elif cfg.SMOKE:
    n_train_s = cfg.SMOKE_TRAIN_BLOCKS * 40
    n_val_s = cfg.SMOKE_VAL_BLOCKS * 40
    n_test_s = cfg.SMOKE_TEST_BLOCKS * 40
    sents = _smoke_sentences(n_train_s + n_val_s + n_test_s)
    n_train_s = min(n_train_s, max(1, len(sents) - 16))
    n_val_s = min(n_val_s, max(1, (len(sents) - n_train_s) // 2))
    n_test_s = min(n_test_s, max(1, len(sents) - n_train_s - n_val_s))
    train_docs, val_docs, test_docs = [], [], []
    spd = cfg.SENTS_PER_DOC
    for i in range(0, n_train_s, spd):
        train_docs.append(" ".join(sents[i:i + spd]))
    for i in range(n_train_s, n_train_s + n_val_s, spd):
        val_docs.append(" ".join(sents[i:i + spd]))
    for i in range(n_train_s + n_val_s, n_train_s + n_val_s + n_test_s, spd):
        test_docs.append(" ".join(sents[i:i + spd]))

    def tok(examples):
        return tokenizer(examples["text"], return_attention_mask=True, add_special_tokens=False)

    def group(examples):
        ids = list(chain(*examples["input_ids"]))
        masks = list(chain(*examples["attention_mask"]))
        nblk = (len(ids) // cfg.BLOCK_SIZE) * cfg.BLOCK_SIZE
        ids, masks = ids[:nblk], masks[:nblk]
        return {
            "input_ids": [ids[i:i + cfg.BLOCK_SIZE] for i in range(0, nblk, cfg.BLOCK_SIZE)],
            "attention_mask": [masks[i:i + cfg.BLOCK_SIZE] for i in range(0, nblk, cfg.BLOCK_SIZE)],
        }

    def _tc(split, desc):
        t = split.map(tok, batched=True, batch_size=tok_bs, remove_columns=["text"], num_proc=1, desc=f"tok_{desc}")
        return t.map(group, batched=True, batch_size=chunk_bs, num_proc=1, desc=f"chunk_{desc}")

    raw_splits = DatasetDict({
        "train": Dataset.from_dict({"text": train_docs}),
        "validation": Dataset.from_dict({"text": val_docs}),
        "test": Dataset.from_dict({"text": test_docs}),
    })
    os.makedirs(chunked_dir, exist_ok=True)
    chunked_train = _tc(raw_splits["train"], "train")
    chunked_val = _tc(raw_splits["validation"], "val")
    chunked_test = _tc(raw_splits["test"], "test")
    if len(chunked_train) > cfg.SMOKE_TRAIN_BLOCKS:
        chunked_train = chunked_train.select(range(cfg.SMOKE_TRAIN_BLOCKS))
    if len(chunked_val) > cfg.SMOKE_VAL_BLOCKS:
        chunked_val = chunked_val.select(range(cfg.SMOKE_VAL_BLOCKS))
    if len(chunked_test) > cfg.SMOKE_TEST_BLOCKS:
        chunked_test = chunked_test.select(range(cfg.SMOKE_TEST_BLOCKS))
    chunked_train.save_to_disk(train_path)
    chunked_val.save_to_disk(val_path)
    chunked_test.save_to_disk(test_path)
    _write_dataset_dict_json(chunked_dir)

else:
    raw = _load_raw()
    n = len(raw)
    n_test = max(1, int(n * cfg.TEST_FRAC))
    n_val = max(1, int(n * cfg.VAL_FRAC))
    n_train = n - n_val - n_test
    assert n_train > 1000, f"train split too small: {n_train} (dataset len={n})"

    os.makedirs(chunked_dir, exist_ok=True)

    val_raw = raw.select(range(n_train, n_train + n_val))
    chunked_val = _materialize_split(val_raw, "val", val_path, sharded=False)
    del val_raw
    _reclaim("val done")

    test_raw = raw.select(range(n_train + n_val, n))
    chunked_test = _materialize_split(test_raw, "test", test_path, sharded=False)
    del test_raw
    _reclaim("test done")

    train_raw = raw.select(range(0, n_train))
    del raw
    _reclaim("raw released; starting train shards")
    chunked_train = _materialize_split(train_raw, "train", train_path, sharded=True)
    del train_raw
    _reclaim("train done")

    assert len(chunked_train) > 100, f"chunked train too small: {len(chunked_train)}"
    _write_dataset_dict_json(chunked_dir)

if os.path.isfile(freqs_path):
    token_freqs = torch.load(freqs_path, map_location="cpu", weights_only=True)
elif os.path.isdir(shard_root):
    token_freqs = count_freqs_from_shards(shard_root, tokenizer.vocab_size)
else:
    token_freqs = count_token_freqs_resumable(chunked_train, tokenizer.vocab_size)
_reclaim("freqs done")

print(
    f"READY {'SMOKE' if cfg.SMOKE else 'PAPER'} | blocks train/val/test: "
    f"{len(chunked_train):,} / {len(chunked_val):,} / {len(chunked_test):,}",
    flush=True,
)
print("active vocab:", int((token_freqs > 0).sum()), "/", tokenizer.vocab_size, flush=True)


DST0 = os.path.join(cfg.OUT, f"gpt2_bookcorpus_seed0_nembd{cfg.N_EMBD}")
DST1 = os.path.join(cfg.OUT, f"gpt2_bookcorpus_seed1_nembd{cfg.N_EMBD}")

for seed, d in ((0, DST0), (1, DST1)):
    if not (os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))):
        raise FileNotFoundError(
            f"Missing base checkpoint for seed {seed}: {d}\n"
            "Run train_seeds.py first to train both base models."
        )

merger_base_dirs = {0: DST0, 1: DST1}
print("merger_base_dirs:", merger_base_dirs, flush=True)


class MatrixType(Enum):
    PERM = "permutation"
    ORTHO = "orthogonal"


class SamplerType(Enum):
    GAUSSIAN = "gaussian"
    UNI = "uniform"
    NARROW_UNI = "narrow_uniform"
    NARROW_UNI_BIASED = "narrow_uniform_biased"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, bias=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        self.weight = nn.Parameter(torch.ones(dim, **kw))
        self.bias = nn.Parameter(torch.zeros(dim, **kw)) if bias else None

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        out = self.weight * (x / rms)
        if self.bias is not None:
            out = out + self.bias
        return out


def absorb_ln_scale(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.ln_1.bias.copy_(block.ln_1.bias / block.ln_1.weight)
            block.attn.c_attn.weight.copy_(block.attn.c_attn.weight * block.ln_1.weight.unsqueeze(1))
            block.ln_1.weight.copy_(torch.ones_like(block.ln_1.weight))
            block.ln_2.bias.copy_(block.ln_2.bias / block.ln_2.weight)
            block.mlp.c_fc.weight.copy_(block.mlp.c_fc.weight * block.ln_2.weight.unsqueeze(1))
            block.ln_2.weight.copy_(torch.ones_like(block.ln_2.weight))
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        model.transformer.ln_f.bias.copy_(model.transformer.ln_f.bias / model.transformer.ln_f.weight)
        model.lm_head.weight.copy_(model.lm_head.weight * model.transformer.ln_f.weight)
        model.transformer.ln_f.weight.copy_(torch.ones_like(model.transformer.ln_f.weight))


def replace_layernorm(module):
    with torch.no_grad():
        for name, child in module.named_children():
            replace_layernorm(child)
            if isinstance(child, nn.LayerNorm):
                dev = child.weight.device
                dt = child.weight.dtype
                rms = RMSNorm(child.normalized_shape, eps=child.eps, bias=True, device=dev, dtype=dt)
                nn.init.ones_(rms.weight)
                if child.bias is not None:
                    rms.bias.copy_(child.bias.to(device=dev, dtype=dt))
                else:
                    rms.bias.zero_()
                setattr(module, name, rms)


def apply_mean_subtraction_to_weights(model):
    ref = model.transformer.h[0].ln_1.bias
    dim = ref.shape[0]
    M = torch.eye(dim, device=ref.device, dtype=ref.dtype) - (
        torch.ones(dim, dim, device=ref.device, dtype=ref.dtype) / dim
    )
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.copy_(block.attn.c_proj.weight @ M)
            block.attn.c_proj.bias.copy_(block.attn.c_proj.bias @ M)
            block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight @ M)
            block.mlp.c_proj.bias.copy_(block.mlp.c_proj.bias @ M)
        model.transformer.wte.weight.copy_(model.transformer.wte.weight @ M)
        model.transformer.wpe.weight.copy_(model.transformer.wpe.weight @ M)


def make_Q(M, N, device=None, dtype=None):
    kw = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    A = torch.randn(M, N, **kw)
    Q, R = torch.linalg.qr(A, mode="reduced")
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs


def expand(model, n_embd_new):
    N_old = model.transformer.wte.weight.shape[1]
    M_new = n_embd_new
    assert N_old < M_new
    c = (N_old / M_new) ** 0.5
    ref = model.transformer.wte.weight
    O = make_Q(M=M_new, N=N_old, device=ref.device, dtype=ref.dtype).t()
    trans_scaled_O = O.t() * c
    with torch.no_grad():
        model.transformer.wte.weight.data = model.transformer.wte.weight @ O
        model.transformer.wpe.weight.data = model.transformer.wpe.weight @ O
        for block in model.transformer.h:
            block.ln_1.bias.data = (O.t() @ block.ln_1.bias) * (1.0 / c)
            block.ln_1.weight.data = torch.ones(M_new, device=ref.device, dtype=ref.dtype)
            block.ln_1.eps *= (N_old / M_new)
            block.attn.c_attn.weight.data = trans_scaled_O @ block.attn.c_attn.weight
            block.attn.c_proj.weight.data = block.attn.c_proj.weight @ O
            block.attn.c_proj.bias.data = block.attn.c_proj.bias @ O
            block.attn.c_proj.nf = M_new
            block.ln_2.bias.data = (O.t() @ block.ln_2.bias) * (1.0 / c)
            block.ln_2.weight.data = torch.ones(M_new, device=ref.device, dtype=ref.dtype)
            block.ln_2.eps *= (N_old / M_new)
            block.mlp.c_fc.weight.data = trans_scaled_O @ block.mlp.c_fc.weight
            block.mlp.c_proj.weight.data = block.mlp.c_proj.weight @ O
            block.mlp.c_proj.bias.data = block.mlp.c_proj.bias @ O
            block.mlp.c_proj.nf = M_new
            block.attn.embed_dim = M_new
        model.transformer.ln_f.bias.data = (O.t() @ model.transformer.ln_f.bias) * (1.0 / c)
        model.transformer.ln_f.weight.data = torch.ones(M_new, device=ref.device, dtype=ref.dtype)
        model.transformer.ln_f.eps *= (N_old / M_new)
        model.lm_head.weight.data = model.lm_head.weight @ O * c
    return model


def _make_orthogonal(A):
    """Closest orthogonal matrix; never crash on NaN / non-convergent SVD."""
    orig_device, orig_dtype = A.device, A.dtype
    A_det = torch.nan_to_num(A.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    A64 = A_det.float().cpu()
    n, m = A64.shape
    if n == m:
        A64 = A64 + 1e-5 * torch.eye(n, dtype=torch.float32)

    try:
        U, _, Vt = torch.linalg.svd(A64, full_matrices=False)
        out = U @ Vt
    except Exception:
        try:
            Q, R = torch.linalg.qr(A64, mode="reduced")
            diag = torch.diag(R) if R.ndim == 2 else R
            signs = torch.sign(diag)
            signs[signs == 0] = 1.0
            out = Q * signs
        except Exception:
            out = torch.eye(n, m, dtype=torch.float32)

    out = out.to(device=orig_device, dtype=orig_dtype)
    return out.detach() + (A - A.detach())


def _make_permutation(P):
    P2 = torch.nan_to_num(P.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    row, col = linear_sum_assignment(-P2.float().cpu().numpy())
    out = torch.zeros_like(P)
    out[row, col] = 1
    return out


def project(A, matrix_type):
    if matrix_type == MatrixType.PERM:
        return _make_permutation(A).detach() + (A - A.detach())
    if matrix_type == MatrixType.ORTHO:
        return _make_orthogonal(A)
    raise ValueError(matrix_type)


def interpolate(W0, W1, coeff):
    return coeff * W0 + (1 - coeff) * W1


def permute_mlp(model, idx, P):
    with torch.no_grad():
        model.transformer.h[idx].mlp.c_fc.weight.copy_(model.transformer.h[idx].mlp.c_fc.weight @ P)
        model.transformer.h[idx].mlp.c_fc.bias.copy_(model.transformer.h[idx].mlp.c_fc.bias @ P)
        model.transformer.h[idx].mlp.c_proj.weight.copy_(P.t() @ model.transformer.h[idx].mlp.c_proj.weight)


def permute_heads(model, layer_idx, P):
    with torch.no_grad():
        def permute(A, P_):
            return torch.matmul(P_, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])

        attn = model.transformer.h[layer_idx].attn
        num_heads, embed_dim = attn.num_heads, attn.embed_dim
        c_attn, c_proj = attn.c_attn, attn.c_proj

        c_attn_weight = torch.cat((c_attn.weight.t(), c_attn.bias.data.view(1, -1).t()), dim=1)
        Q, K, V = c_attn_weight.data.chunk(3, dim=0)
        Q = rearrange(Q, "(h d) m -> h d m", h=num_heads, m=embed_dim + 1)
        K = rearrange(K, "(h d) m -> h d m", h=num_heads, m=embed_dim + 1)
        V = rearrange(V, "(h d) m -> h d m", h=num_heads, m=embed_dim + 1)
        Q, K, V = permute(Q, P), permute(K, P), permute(V, P)

        OUT = rearrange(c_proj.weight.data.t(), " m (h d) -> m h d", h=num_heads, m=embed_dim)
        OUT = OUT.permute(1, 2, 0)
        OUT = permute(OUT, P)

        QK = torch.bmm(Q.transpose(1, 2), K)
        OUTV = OUT.transpose(1, 2) @ V
        _dev, _dt = QK.device, QK.dtype

        Q_new = torch.zeros(QK.shape, device=_dev, dtype=_dt)
        K_new = torch.zeros(QK.shape, device=_dev, dtype=_dt)
        V_new = torch.zeros(QK.shape, device=_dev, dtype=_dt)
        OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1], device=_dev, dtype=_dt)

        for h in range(QK.size(0)):
            eye = torch.eye(QK.shape[1], device=_dev, dtype=_dt)
            Q_new[h] = QK[h].t()
            K_new[h] = eye
            OUT_new[h] = OUTV[h].t()
            V_new[h] = eye

        Q_new = Q_new.reshape(-1, embed_dim + 1)
        K_new = K_new.reshape(-1, embed_dim + 1)
        V_new = V_new.reshape(-1, embed_dim + 1)
        new_w = torch.cat((Q_new, K_new, V_new), dim=0)[:, :-1].t().contiguous()
        new_b = torch.cat((Q_new, K_new, V_new), dim=0)[:, -1].contiguous()
        OUT_new = OUT_new.permute(2, 0, 1).reshape(embed_dim, -1).t().contiguous()

        L = model.transformer.h[layer_idx].attn
        L.c_attn.nx = new_w.shape[0]
        L.c_attn.nf = new_w.shape[1]
        L.c_attn.weight = nn.Parameter(new_w.clone())
        L.c_attn.bias = nn.Parameter(new_b.clone())
        L.c_proj.nx = OUT_new.shape[0]
        L.c_proj.nf = OUT_new.shape[1]
        L.c_proj.weight = nn.Parameter(OUT_new.clone())
        L.split_size = new_w.shape[1] // 3
        L.head_dim = L.embed_dim + 1


def project_to_attn_circuits(model, layer_idx):
    with torch.no_grad():
        attn = model.transformer.h[layer_idx].attn
        num_heads, embed_dim = attn.num_heads, attn.embed_dim
        c_attn, c_proj = attn.c_attn, attn.c_proj
        c_attn_weight = torch.cat((c_attn.weight.t(), c_attn.bias.data.view(1, -1).t()), dim=1)
        Q, K, V = c_attn_weight.data.chunk(3, dim=0)
        Q = rearrange(Q, "(h d) m -> h d m", h=num_heads, m=embed_dim + 1)
        K = rearrange(K, "(h d) m -> h d m", h=num_heads, m=embed_dim + 1)
        V = rearrange(V, "(h d) m -> h d m", h=num_heads, m=embed_dim + 1)
        OUT = rearrange(c_proj.weight.data.t(), " m (h d) -> m h d", h=num_heads, m=embed_dim)
        OUT = OUT.permute(1, 2, 0)
        QK = torch.bmm(Q.transpose(1, 2), K)
        OUTV = OUT.transpose(1, 2) @ V

        _dev, _dt = QK.device, QK.dtype
        Q_new = torch.zeros(QK.shape, device=_dev, dtype=_dt)
        K_new = torch.zeros(QK.shape, device=_dev, dtype=_dt)
        V_new = torch.zeros(QK.shape, device=_dev, dtype=_dt)
        OUT_new = torch.zeros(OUTV.shape[0], OUTV.shape[2], OUTV.shape[1], device=_dev, dtype=_dt)

        scale = (QK.shape[1] ** 0.5) / (V.shape[1] ** 0.5)
        for h in range(QK.size(0)):
            eye = torch.eye(QK.shape[1], device=_dev, dtype=_dt)
            Q_new[h] = QK[h].t() * scale
            K_new[h] = eye
            OUT_new[h] = OUTV[h].t()
            V_new[h] = eye

        Q_new = Q_new.reshape(-1, embed_dim + 1)
        K_new = K_new.reshape(-1, embed_dim + 1)
        V_new = V_new.reshape(-1, embed_dim + 1)
        new_w = torch.cat((Q_new, K_new, V_new), dim=0)[:, :-1].t().contiguous()
        new_b = torch.cat((Q_new, K_new, V_new), dim=0)[:, -1].contiguous()
        OUT_new = OUT_new.permute(2, 0, 1).reshape(embed_dim, -1).t().contiguous()

        L = model.transformer.h[layer_idx].attn
        L.c_attn.nx = new_w.shape[0]
        L.c_attn.nf = new_w.shape[1]
        L.c_attn.weight = nn.Parameter(new_w.clone())
        L.c_attn.bias = nn.Parameter(new_b.clone())
        L.c_proj.nx = OUT_new.shape[0]
        L.c_proj.nf = OUT_new.shape[1]
        L.c_proj.weight = nn.Parameter(OUT_new.clone())
        L.split_size = new_w.shape[1] // 3
        L.head_dim = L.embed_dim + 1


def ortho_residual(model, O):
    with torch.no_grad():
        O = O.to(dtype=model.transformer.wte.weight.dtype, device=model.transformer.wte.weight.device)
        model.transformer.wte.weight.copy_(model.transformer.wte.weight @ O)
        model.transformer.wpe.weight.copy_(model.transformer.wpe.weight @ O)
        for block in model.transformer.h:
            block.ln_1.bias.copy_(O.t() @ block.ln_1.bias)
            block.attn.c_attn.weight.copy_(O.t() @ block.attn.c_attn.weight)
            block.attn.c_proj.weight.copy_(block.attn.c_proj.weight @ O)
            block.attn.c_proj.bias.copy_(block.attn.c_proj.bias @ O)
            block.ln_2.bias.copy_(O.t() @ block.ln_2.bias)
            block.mlp.c_fc.weight.copy_(O.t() @ block.mlp.c_fc.weight)
            block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight @ O)
            block.mlp.c_proj.bias.copy_(block.mlp.c_proj.bias @ O)
        model.transformer.ln_f.bias.copy_(O.t() @ model.transformer.ln_f.bias)
        model.lm_head.weight.copy_(model.lm_head.weight @ O)


def compute_optimal_orthogonal_matrix(t1, t2):
    C = t2.T @ t1
    try:
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)
        return U @ Vh
    except Exception:
        C_cpu = C.detach().to("cpu", dtype=torch.float32)
        U, _, Vh = torch.linalg.svd(C_cpu, full_matrices=False)
        return (U @ Vh).to(device=C.device, dtype=C.dtype)


def get_cost_heads(t0, t1, heads):
    a = t0.reshape(heads, -1).float()
    b = t1.reshape(heads, -1).float()
    return torch.cdist(a, b, p=2)


def otify(cost):
    dev = cost.device
    cost_cpu = cost.detach().to("cpu", dtype=torch.float64).contiguous()
    cost_cpu = torch.nan_to_num(cost_cpu, nan=1e6, posinf=1e6, neginf=1e6)
    n = cost_cpu.shape[0]
    a = torch.ones(n, dtype=torch.float64) / n
    b = torch.ones(n, dtype=torch.float64) / n
    m = ot.emd(a, b, cost_cpu)
    return (m * n).to(device=dev, dtype=torch.float32)


def _ot_cost_matrix(X, Y, metric="euclidean2", eps=1e-8):
    if metric == "euclidean2":
        X2 = (X ** 2).sum(dim=0, keepdim=True)
        Y2 = (Y ** 2).sum(dim=0, keepdim=True)
        C = X2.T + Y2 - 2 * (X.T @ Y)
        return torch.clamp(C, min=0)
    if metric == "cosine":
        Xn = X / (X.norm(dim=0, keepdim=True) + eps)
        Yn = Y / (Y.norm(dim=0, keepdim=True) + eps)
        return 1.0 - (Xn.T @ Yn)
    raise ValueError(metric)


def compute_optimal_permutation_matrix_ot(t1, t2, metric="euclidean2"):
    N, M = t1.shape
    C = _ot_cost_matrix(t1, t2, metric=metric).detach().cpu().numpy()
    a, b = ot.unif(M), ot.unif(M)
    T = ot.emd(a, b, C)
    P = torch.from_numpy(T).to(device=t1.device, dtype=t1.dtype)
    P = (P * M).round()
    return P, None, None, None


def weight_matching(model0, model1, heads, iterations=15, permutations_only=False,
                    token_freqs=None, block_size=None, topk_vocab=None):
    device = next(model0.parameters()).device
    active = None
    if token_freqs is not None:
        freqs = token_freqs.detach().to("cpu")
        if topk_vocab is not None and int(topk_vocab) < freqs.numel():
            active = torch.topk(freqs, k=int(topk_vocab)).indices
            active = active[freqs[active] > 0]
            active, _ = torch.sort(active)
        else:
            active = (freqs > 0).nonzero(as_tuple=False).flatten()
        active = active.to(device=device, dtype=torch.long)
        print(f"  WM active vocab: {active.numel():,}", flush=True)

    for it in range(iterations):
        tok0 = model0.transformer.wte.weight.data
        tok1 = model1.transformer.wte.weight.data
        if active is not None:
            tok0 = tok0.index_select(0, active)
            tok1 = tok1.index_select(0, active)

        pos0 = model0.transformer.wpe.weight.data
        pos1 = model1.transformer.wpe.weight.data
        if block_size is not None:
            bs = int(block_size)
            pos0 = pos0[:bs]
            pos1 = pos1[:bs]

        head0 = model0.lm_head.weight.data
        head1 = model1.lm_head.weight.data
        if active is not None:
            head0 = head0.index_select(0, active)
            head1 = head1.index_select(0, active)

        layers_0 = [tok0.t(), pos0.t(), head0.t()]
        layers_1 = [tok1.t(), pos1.t(), head1.t()]
        if it > 0:
            for li in range(len(model1.transformer.h)):
                layers_0.append(model0.transformer.h[li].attn.c_attn.weight.data)
                layers_1.append(model1.transformer.h[li].attn.c_attn.weight.data)
                layers_0.append(model0.transformer.h[li].attn.c_proj.weight.data.t())
                layers_1.append(model1.transformer.h[li].attn.c_proj.weight.data.t())
                layers_0.append(model0.transformer.h[li].mlp.c_fc.weight.data)
                layers_1.append(model1.transformer.h[li].mlp.c_fc.weight.data)
                layers_0.append(model0.transformer.h[li].mlp.c_proj.weight.data.t())
                layers_1.append(model1.transformer.h[li].mlp.c_proj.weight.data.t())

        layers_0 = [l / (l.shape[1] ** 0.5) for l in layers_0]
        layers_1 = [l / (l.shape[1] ** 0.5) for l in layers_1]

        if permutations_only:
            O, *_ = compute_optimal_permutation_matrix_ot(
                torch.cat(layers_0, dim=1).t(), torch.cat(layers_1, dim=1).t()
            )
            O = O.t()
        else:
            O = compute_optimal_orthogonal_matrix(
                torch.cat(layers_0, dim=1).t(), torch.cat(layers_1, dim=1).t()
            )
        ortho_residual(model1, O)

        for li in range(len(model1.transformer.h)):
            def get_qkv(model, layer_i=li):
                attn = model.transformer.h[layer_i].attn
                nh, ed = attn.num_heads, attn.embed_dim
                caw = torch.cat((attn.c_attn.weight.t(), attn.c_attn.bias.data.view(1, -1).t()), dim=1)
                Q, K, V = caw.data.chunk(3, dim=0)
                Q = rearrange(Q, "(h d) m -> h d m", h=nh, m=ed + 1)
                K = rearrange(K, "(h d) m -> h d m", h=nh, m=ed + 1)
                V = rearrange(V, "(h d) m -> h d m", h=nh, m=ed + 1)
                OUT = rearrange(attn.c_proj.weight.data.t(), " m (h d) -> m h d", h=nh, m=ed)
                OUT = OUT.permute(1, 2, 0)
                return torch.bmm(Q.transpose(1, 2), K), OUT.transpose(1, 2) @ V

            QK0, OUTV0 = get_qkv(model0)
            QK1, OUTV1 = get_qkv(model1)
            cost = get_cost_heads(QK0, QK1, heads=heads) + get_cost_heads(OUTV0, OUTV1, heads=heads)
            del QK0, OUTV0, QK1, OUTV1
            P = otify(cost).to(device)
            permute_heads(model1, li, P)

            ff0 = torch.cat((
                model0.transformer.h[li].mlp.c_fc.weight.data.t(),
                model0.transformer.h[li].mlp.c_fc.bias.unsqueeze(1),
                model0.transformer.h[li].mlp.c_proj.weight.data,
            ), dim=1)
            ff1 = torch.cat((
                model1.transformer.h[li].mlp.c_fc.weight.data.t(),
                model1.transformer.h[li].mlp.c_fc.bias.unsqueeze(1),
                model1.transformer.h[li].mlp.c_proj.weight.data,
            ), dim=1)
            n0 = torch.norm(ff0, dim=-1, keepdim=True).clamp(min=1e-8)
            n1 = torch.norm(ff1, dim=-1, keepdim=True).clamp(min=1e-8)
            cost_ff = torch.cdist(ff0 / n0, ff1 / n1, p=1)
            P_ff = otify(cost_ff).to(device)
            permute_mlp(model1, li, P_ff.t())
            del ff0, ff1, cost_ff, P_ff, cost, P

        print(f"  weight_matching iter {it + 1}/{iterations}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model1


class StageLog:
    def __init__(self, path, prefix="STAGE"):
        self.path = path
        self.prefix = prefix
        self.t0 = time.time()
        self.last = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _vram(self):
        if not torch.cuda.is_available():
            return "cpu"
        free, total = torch.cuda.mem_get_info()
        return f"VRAM free {free/1e9:.1f}/{total/1e9:.1f} GB"

    def mark(self, stage, ok=True, err=None):
        elapsed = time.time() - self.t0
        status = "OK" if ok else "FAIL"
        line = f"[{self.prefix}] {status} | t+{elapsed:.1f}s | {stage} | {self._vram()}"
        if err is not None:
            line += f" | {type(err).__name__}: {err}"
        print(line, flush=True)
        self.last = stage
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
            except Exception:
                pass

    def run(self, stage, fn):
        self.mark(f"ENTER {stage}")
        try:
            out = fn()
            self.mark(f"EXIT  {stage}")
            return out
        except Exception as e:
            self.mark(f"CRASH {stage}", ok=False, err=e)
            traceback.print_exc()
            raise


def _clone_buf(t):
    return t.detach().clone().contiguous()


class Conv1DMerger(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.register_buffer("w0", _clone_buf(c0.weight.data))
        self.register_buffer("w1", _clone_buf(c1.weight.data))
        if c0.bias is not None and c1.bias is not None:
            self.register_buffer("b0", _clone_buf(c0.bias.data))
            self.register_buffer("b1", _clone_buf(c1.bias.data))
        else:
            self.b0 = None
            self.b1 = None
        self.P_in = None
        self.P_out = None
        self.nf = c0.nf
        self.coeff = None

    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P
    def set_coeff(self, c): self.coeff = c

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.w0, self.P_in @ self.w1 @ self.P_out, self.coeff)
        bias = None
        if self.b0 is not None and self.b1 is not None:
            bias = interpolate(self.b0, self.b1 @ self.P_out, self.coeff)
        x = x.view(-1, x.size(-1))
        x = torch.addmm(bias, x, weight) if bias is not None else torch.matmul(x, weight)
        return x.view(size_out)


class LinearMerger(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.register_buffer("w0", _clone_buf(c0.weight.data.t()))
        self.register_buffer("w1", _clone_buf(c1.weight.data.t()))
        if c0.bias is not None and c1.bias is not None:
            self.register_buffer("b0", _clone_buf(c0.bias.data))
            self.register_buffer("b1", _clone_buf(c1.bias.data))
        else:
            self.b0 = None
            self.b1 = None
        self.P_in = None
        self.P_out = None
        self.nf = self.w0.shape[1]
        self.nx = self.w0.shape[0]
        self.coeff = None

    def set_coeff(self, c): self.coeff = c
    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        weight = interpolate(self.w0, self.P_in @ self.w1, self.coeff)
        bias = None
        if self.b0 is not None and self.b1 is not None:
            b1 = self.b1 if self.P_out is None else (self.b1 @ self.P_out)
            bias = interpolate(self.b0, b1, self.coeff)
        x = x.view(-1, x.size(-1))
        x = torch.addmm(bias, x, weight) if bias is not None else torch.matmul(x, weight)
        return x.view(size_out)


class Conv1DMergerCATTN(nn.Module):
    def __init__(self, c0, c1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("w0", _clone_buf(c0.weight.data))
        self.register_buffer("b0", _clone_buf(c0.bias.data))
        self.register_buffer("w1", _clone_buf(c1.weight.data))
        self.register_buffer("b1", _clone_buf(c1.bias.data))
        self.P_in = None
        self.P_out = None
        self.nf = c0.nf
        self.nx = c0.nx
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coeff = None

    def set_coeff(self, c): self.coeff = c
    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P

    def _permute_heads(self, weight, bias, P):
        def permute(A, P_):
            return torch.matmul(P_, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        caw = torch.cat((weight.t(), bias.reshape(-1, 1)), dim=1)
        Q, K, V = caw.chunk(3, dim=0)
        Q = torch.cat((Q[:, :-1] @ self.P_in.t(), Q[:, -1:].contiguous()), dim=-1)
        Q = rearrange(Q, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim + 1)
        K = rearrange(K, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim + 1)
        V = rearrange(V, "(h d) m -> h d m", h=self.num_heads, m=self.embed_dim + 1)
        Q = torch.cat((
            torch.bmm(Q.transpose(1, 2)[:, :, :-1], self.P_in.t().expand(self.num_heads, -1, -1)),
            Q.transpose(1, 2)[:, :, -1:],
        ), dim=-1).transpose(1, 2)
        Q, K, V = permute(Q, P), permute(K, P), permute(V, P)
        Q = Q.reshape(-1, self.embed_dim + 1)
        K = K.reshape(-1, self.embed_dim + 1)
        V = V.reshape(-1, self.embed_dim + 1)
        cat = torch.cat((Q, K, V), dim=0)
        return cat[:, :-1].t().contiguous(), cat[:, -1].contiguous()

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        w1, b1 = self._permute_heads(self.w1, self.b1, self.P_out)
        bias = interpolate(self.b0, b1, self.coeff)
        weight = interpolate(self.w0, w1, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        return x.view(size_out)


class Conv1DMergerCPROJ(nn.Module):
    def __init__(self, c0, c1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("w0", _clone_buf(c0.weight.data))
        self.register_buffer("b0", _clone_buf(c0.bias.data))
        self.register_buffer("w1", _clone_buf(c1.weight.data))
        self.register_buffer("b1", _clone_buf(c1.bias.data))
        self.P_in = None
        self.P_out = None
        self.nf = c0.nf
        self.nx = c0.nx
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coeff = None

    def set_coeff(self, c): self.coeff = c
    def set_P_in(self, P): self.P_in = P
    def set_P_out(self, P): self.P_out = P

    def _permute_heads(self, weight, bias, P):
        def permute(A, P_):
            return torch.matmul(P_, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        OUT = rearrange(weight.t(), " m (h d) -> m h d", h=self.num_heads, m=self.embed_dim)
        OUT = OUT.permute(1, 2, 0)
        OUT = torch.cat((
            OUT.transpose(1, 2)[:, :, :-1] @ self.P_out.expand(self.num_heads, -1, -1),
            OUT.transpose(1, 2)[:, :, -1:],
        ), dim=-1).transpose(1, 2)
        OUT = permute(OUT, P)
        OUT = OUT.permute(2, 0, 1).reshape(self.embed_dim, -1).t().contiguous()
        return OUT, bias

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        w1, b1 = self._permute_heads(self.w1 @ self.P_out, self.b1 @ self.P_out, self.P_in)
        bias = interpolate(self.b0, b1, self.coeff)
        weight = interpolate(self.w0, w1, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        return x.view(size_out)


class RMSMerger(nn.Module):
    def __init__(self, r0, r1):
        super().__init__()
        self.register_buffer("bias_0", _clone_buf(r0.bias.data))
        self.register_buffer("bias_1", _clone_buf(r1.bias.data))
        dim = int(r0.weight.shape[0])
        self.norm = RMSNorm(dim=dim, eps=r0.eps, bias=False, device=r0.weight.device, dtype=r0.weight.dtype)
        with torch.no_grad():
            self.norm.weight.fill_(1.0)
        self.norm.weight.requires_grad_(False)
        self.P = None
        self.coeff = None

    def set_coeff(self, c): self.coeff = c
    def set_P(self, P): self.P = P

    def forward(self, x):
        x = self.norm(x)
        return x + interpolate(self.bias_0, self.P @ self.bias_1, coeff=self.coeff)


class EmbeddingMerger(nn.Module):
    def __init__(self, e0, e1):
        super().__init__()
        self.num_embeddings = e0.num_embeddings
        self.embedding_dim = e0.embedding_dim
        self.register_buffer("w0", _clone_buf(e0.weight.data))
        self.register_buffer("w1", _clone_buf(e1.weight.data))
        self.P = None
        self.coeff = None

    def set_coeff(self, c): self.coeff = c
    def set_P(self, P): self.P = P

    def forward(self, x):
        e0 = torch.nn.functional.embedding(x, self.w0)
        e1 = torch.nn.functional.embedding(x, self.w1)
        return interpolate(e0, e1 @ self.P, coeff=self.coeff)


class GPTMerger(nn.Module):
    def _absorb(self, model):
        model.eval()
        try:
            model.config.attn_implementation = "eager"
            model._attn_implementation = "eager"
        except Exception:
            pass

        prev = next(model.parameters()).device
        model.cpu()
        gc.collect()

        bs, sl = 2, 32
        vocab = model.lm_head.weight.shape[0]
        dummy = torch.randint(0, vocab, (bs, sl))

        with torch.no_grad():
            o0 = model(input_ids=dummy).logits

        absorb_ln_scale(model)
        replace_layernorm(model)
        apply_mean_subtraction_to_weights(model)

        with torch.no_grad():
            o1 = model(input_ids=dummy).logits

        max_abs = (o0 - o1).abs().max().item()
        mean_abs = (o0 - o1).abs().mean().item()
        print(f"[absorb] max|d|={max_abs:.3e} mean|d|={mean_abs:.3e}", flush=True)
        assert max_abs < 5e-5, f"absorb changed outputs! max|d|={max_abs:.3e}"

        model.to(prev)
        if prev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def __init__(self, model0, model1, token_freqs=None, permutations_only=False,
                 iterations=15, topk_vocab=None, match_device=None, stage_log=None):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("GPTMerger requires CUDA")
        md = torch.device(match_device) if match_device is not None else torch.device("cuda")
        if md.type != "cuda":
            raise RuntimeError(f"match_device must be CUDA, got {md}")
        log = stage_log or StageLog(None, prefix="GPTMerger")

        def step(name, fn):
            return log.run(name, fn)

        model0 = model0.eval().to(md)
        model1 = model1.eval().to(md)
        if token_freqs is not None:
            token_freqs = token_freqs.to(md)
        log.mark("models_on_cuda")

        step("absorb_model0", lambda: self._absorb(model0))
        torch.cuda.empty_cache()
        step("absorb_model1", lambda: self._absorb(model1))
        torch.cuda.empty_cache()
        self._permutations_only = permutations_only

        def random_parameter(d0, d1=None):
            d1 = d0 if d1 is None else d1
            eye = torch.eye(d0, d1, device=md)
            return nn.Parameter(eye + torch.randn_like(eye) * 1e-2)

        embed_dim = model0.transformer.wte.weight.shape[1]
        num_heads = model0.transformer.h[0].attn.num_heads
        n_layers = len(model0.transformer.h)

        def _circuits():
            for i in range(n_layers):
                project_to_attn_circuits(model0, i)
            for i in range(n_layers):
                project_to_attn_circuits(model1, i)
        step("project_attn_circuits", _circuits)

        assert model0.transformer.wte.weight.shape[1] >= model1.transformer.wte.weight.shape[1]
        if model0.transformer.wte.weight.shape[1] > model1.transformer.wte.weight.shape[1]:
            assert not permutations_only

            def _expand():
                nonlocal model1
                model1 = expand(model1, model0.transformer.wte.weight.shape[1])
                for i in range(len(model1.transformer.h)):
                    project_to_attn_circuits(model1, i)
            step("expand_model1", _expand)

        def _wm():
            weight_matching(
                model0, model1, heads=num_heads, iterations=iterations,
                token_freqs=token_freqs, permutations_only=permutations_only,
                topk_vocab=topk_vocab, block_size=getattr(model0.config, "n_positions", None),
            )
            torch.cuda.synchronize()
        step(f"weight_matching_{iterations}iters_perm={permutations_only}", _wm)

        log.mark("move_matched_models_to_cpu")
        model0_cpu = model0.cpu()
        model1_cpu = model1.cpu()
        del model0, model1, token_freqs
        gc.collect()
        torch.cuda.empty_cache()

        self.proj = nn.ParameterDict({
            "residual": random_parameter(
                model0_cpu.transformer.wte.weight.shape[1],
                model1_cpu.transformer.wte.weight.shape[1],
            )
        })
        for i in range(n_layers):
            self.proj[f"attention_heads_{i}"] = random_parameter(num_heads)
            self.proj[f"mlp_{i}"] = random_parameter(model0_cpu.transformer.h[i].mlp.c_fc.bias.shape[0])

        def _build_modules():
            model_cfg = copy.deepcopy(model0_cpu.config)
            try:
                model_cfg.attn_implementation = "eager"
            except Exception:
                pass
            shell = GPT2LMHeadModel(model_cfg)
            shell.transformer.wte = EmbeddingMerger(model0_cpu.transformer.wte, model1_cpu.transformer.wte)
            shell.transformer.wpe = EmbeddingMerger(model0_cpu.transformer.wpe, model1_cpu.transformer.wpe)
            for i in range(n_layers):
                log.mark(f"wire_layer_{i}")
                h0 = model0_cpu.transformer.h[i]
                h1 = model1_cpu.transformer.h[i]
                shell.transformer.h[i].ln_1 = RMSMerger(h0.ln_1, h1.ln_1)
                shell.transformer.h[i].attn.c_attn = Conv1DMergerCATTN(
                    h0.attn.c_attn, h1.attn.c_attn, num_heads=num_heads, embed_dim=embed_dim
                )
                shell.transformer.h[i].attn.c_proj = Conv1DMergerCPROJ(
                    h0.attn.c_proj, h1.attn.c_proj, num_heads=num_heads, embed_dim=embed_dim
                )
                shell.transformer.h[i].attn.embed_dim = embed_dim
                shell.transformer.h[i].attn.num_heads = num_heads
                _nf = int(getattr(h0.attn.c_attn, "nf", None) or h0.attn.c_attn.weight.shape[1])
                shell.transformer.h[i].attn.split_size = _nf // 3
                shell.transformer.h[i].attn.head_dim = embed_dim + 1
                shell.transformer.h[i].mlp.c_fc = Conv1DMerger(h0.mlp.c_fc, h1.mlp.c_fc)
                shell.transformer.h[i].mlp.c_proj = Conv1DMerger(h0.mlp.c_proj, h1.mlp.c_proj)
                shell.transformer.h[i].ln_2 = RMSMerger(h0.ln_2, h1.ln_2)
            shell.transformer.ln_f = RMSMerger(model0_cpu.transformer.ln_f, model1_cpu.transformer.ln_f)
            shell.lm_head = LinearMerger(model0_cpu.lm_head, model1_cpu.lm_head)
            self.model = shell

        step("build_merger_modules", _build_modules)
        del model0_cpu, model1_cpu
        gc.collect()
        self.set_sampler(sampler_type=None)
        step("merger_to_cuda", lambda: self.to(md))
        log.mark(f"READY on {next(self.parameters()).device}")

    def set_sampler(self, sampler_type, fixed_coeff=0.5):
        if sampler_type is None:
            self._sampler = lambda: fixed_coeff
            return
        st = str(sampler_type).lower()
        if "narrow_uniform_biased" in st:
            self._sampler = lambda: random.uniform(0.2, 0.5)
        elif "narrow_uniform" in st:
            self._sampler = lambda: random.uniform(0.4, 0.6)
        elif "uniform" in st:
            self._sampler = lambda: random.uniform(0.0, 1.0)
        elif "gaussian" in st:
            self._sampler = lambda: min(max(random.gauss(0.5, 0.1), 0.0), 1.0)
        else:
            self._sampler = lambda: 0.5

    def _project(self, coeff):
        P_res = project(self.proj["residual"],
                        matrix_type=MatrixType.PERM if self._permutations_only else MatrixType.ORTHO)
        self.model.transformer.wte.set_P(P_res)
        self.model.transformer.wte.set_coeff(coeff)
        self.model.transformer.wpe.set_P(P_res)
        self.model.transformer.wpe.set_coeff(coeff)
        for i in range(len(self.model.transformer.h)):
            self.model.transformer.h[i].ln_1.set_P(P_res.t())
            self.model.transformer.h[i].ln_1.set_coeff(coeff)
            self.model.transformer.h[i].attn.c_attn.set_P_in(P_res.t())
            self.model.transformer.h[i].attn.c_attn.set_coeff(coeff)
            P_heads = project(self.proj[f"attention_heads_{i}"], matrix_type=MatrixType.PERM)
            self.model.transformer.h[i].attn.c_attn.set_P_out(P_heads)
            self.model.transformer.h[i].attn.c_proj.set_P_out(P_res)
            self.model.transformer.h[i].attn.c_proj.set_P_in(P_heads)
            self.model.transformer.h[i].attn.c_proj.set_coeff(coeff)
            self.model.transformer.h[i].mlp.c_fc.set_P_in(P_res.t())
            P_mlp = project(self.proj[f"mlp_{i}"], matrix_type=MatrixType.PERM)
            self.model.transformer.h[i].mlp.c_fc.set_P_out(P_mlp)
            self.model.transformer.h[i].mlp.c_fc.set_coeff(coeff)
            self.model.transformer.h[i].mlp.c_proj.set_P_out(P_res)
            self.model.transformer.h[i].mlp.c_proj.set_P_in(P_mlp.t())
            self.model.transformer.h[i].mlp.c_proj.set_coeff(coeff)
            self.model.transformer.h[i].ln_2.set_P(P_res.t())
            self.model.transformer.h[i].ln_2.set_coeff(coeff)
        self.model.transformer.ln_f.set_P(P_res.t())
        self.model.transformer.ln_f.set_coeff(coeff)
        self.model.lm_head.set_P_in(P_res.t())
        self.model.lm_head.set_coeff(coeff)

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kw):
        coeff = self._sampler()
        self._project(coeff=coeff)
        device = next(self.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)


class GPTMergerWrapper(nn.Module):
    def __init__(self, config, merger_model):
        super().__init__()
        self.merger_model = merger_model
        self.config = config

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kw):
        return self.merger_model.forward(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kw)

    def to(self, device):
        self.merger_model = self.merger_model.to(device)
        return super().to(device)

    def state_dict(self, *a, **k):
        return self.merger_model.state_dict(*a, **k)

    def load_state_dict(self, sd, strict=True):
        return self.merger_model.load_state_dict(sd, strict=strict)


print("Merger classes ready.")


_tr_sig = inspect.signature(Trainer.__init__).parameters
_TOK_KW = "processing_class" if "processing_class" in _tr_sig else "tokenizer"
_ta_sig = inspect.signature(TrainingArguments.__init__).parameters
_eval_kw = "eval_strategy" if "eval_strategy" in _ta_sig else "evaluation_strategy"

status_path = os.path.join(cfg.OUT, "pipeline_status.log")
merger_logs_root = os.path.join(cfg.OUT, "merger_logs")
os.makedirs(merger_logs_root, exist_ok=True)
pipe = StageLog(status_path, prefix="PIPELINE")


def make_train_args(**kwargs):
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in allowed})


def make_trainer(**kw):
    tok = kw.pop("tokenizer", None)
    if tok is not None:
        kw[_TOK_KW] = tok
    return Trainer(**kw)


def load_base(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path).eval()
    try:
        model.config.attn_implementation = "eager"
        model._attn_implementation = "eager"
    except Exception:
        pass
    return model


def _fix_circuit_attn_meta(wrapper):
    emb = int(getattr(wrapper.config, "n_embd", None) or wrapper.config.hidden_size)
    for block in wrapper.merger_model.model.transformer.h:
        nf = int(block.attn.c_attn.nf)
        if nf % 3 != 0:
            raise RuntimeError(f"c_attn.nf={nf} not divisible by 3")
        block.attn.embed_dim = emb
        block.attn.split_size = nf // 3
        block.attn.head_dim = emb + 1
    return wrapper


def build_merger(permutations_only, tag=""):
    label = f"build_merger perm={permutations_only}" + (f" ({tag})" if tag else "")
    slog = StageLog(status_path, prefix=f"MERGER[{tag or ('perm' if permutations_only else 'ortho')}]")

    def _do():
        m0 = load_base(merger_base_dirs[cfg.SEEDS[0]])
        m1 = load_base(merger_base_dirs[cfg.SEEDS[1]])
        conf = m0.config
        try:
            conf.attn_implementation = "eager"
        except Exception:
            pass
        merger = GPTMerger(
            m0, m1, token_freqs=token_freqs, permutations_only=permutations_only,
            iterations=cfg.WEIGHT_MATCH_ITERS, topk_vocab=getattr(cfg, "MATCH_TOPK_VOCAB", None),
            match_device=torch.device("cuda"), stage_log=slog,
        )
        del m0, m1
        _reclaim("after_wm")
        wrapper = GPTMergerWrapper(config=conf, merger_model=merger).cuda()
        return _fix_circuit_attn_meta(wrapper)

    return pipe.run(label, _do)


def state_path_for(tag):
    return os.path.join(cfg.OUT, f"merger_{tag}", "merger_state.pt")


def find_latest_checkpoint(out_dir):
    if not os.path.isdir(out_dir):
        return None
    best, best_n = None, -1
    for name in os.listdir(out_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        n = int(m.group(1))
        path = os.path.join(out_dir, name)
        if os.path.isdir(path) and n > best_n:
            best, best_n = path, n
    return best


class LossLoggerCallback(TrainerCallback):
    def __init__(self, jsonl_path, tag, append=False):
        self.jsonl_path = jsonl_path
        self.tag = tag
        self.history = []
        if not (append and os.path.isfile(jsonl_path)):
            open(self.jsonl_path, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"tag": self.tag, "step": int(state.global_step),
               "epoch": float(state.epoch) if state.epoch is not None else None}
        for key in ("loss", "learning_rate", "grad_norm"):
            if key in logs:
                try:
                    row[key] = float(logs[key])
                except Exception:
                    pass
        self.history.append(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def save_summary(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tag": self.tag, "history": self.history}, f, indent=2)


def train_merger(permutations_only, tag, sampler_type="narrow_uniform"):
    """Train (or resume, or reload) a merger with a differentiable coeff sampler."""
    out_dir = os.path.join(cfg.OUT, f"merger_{tag}")
    state_path = state_path_for(tag)
    jsonl_path = os.path.join(merger_logs_root, f"merger_{tag}_train.jsonl")
    summary_path = os.path.join(merger_logs_root, f"merger_{tag}_train.json")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(state_path):
        print(f"Final state exists: {state_path} — loading (delete to retrain).", flush=True)
        wrapper = build_merger(permutations_only, tag=f"reload_{tag}")
        try:
            sd = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(state_path, map_location="cpu")
        wrapper.load_state_dict(sd)
        _fix_circuit_attn_meta(wrapper)
        return wrapper.cuda()

    resume_ckpt = find_latest_checkpoint(out_dir)
    if resume_ckpt:
        print(f"*** RESUME from {resume_ckpt} ***", flush=True)

    def _train():
        _reclaim(f"before_train_{tag}")
        wrapper = build_merger(permutations_only, tag=tag).cuda()
        _fix_circuit_attn_meta(wrapper)
        wrapper.merger_model.set_sampler(sampler_type=sampler_type)

        loss_cb = LossLoggerCallback(jsonl_path, tag=tag, append=bool(resume_ckpt))
        ta_kwargs = dict(
            output_dir=out_dir,
            logging_strategy="steps", logging_steps=1 if cfg.SMOKE else 25,
            save_strategy="steps", save_steps=int(getattr(cfg, "SAVE_STEPS", 200)),
            save_total_limit=2,
            num_train_epochs=float(cfg.MERGE_EPOCHS),
            per_device_train_batch_size=int(cfg.MERGE_BS),
            per_device_eval_batch_size=int(cfg.EVAL_BS),
            gradient_accumulation_steps=int(cfg.MERGE_GRAD_ACCUM),
            learning_rate=cfg.LR, lr_scheduler_type="cosine",
            warmup_ratio=cfg.WARMUP_RATIO, weight_decay=cfg.WEIGHT_DECAY,
            fp16=False, bf16=True, report_to="none",
            dataloader_num_workers=0, dataloader_pin_memory=False,
            remove_unused_columns=False, no_cuda=False,
        )
        if "tf32" in _ta_sig:
            ta_kwargs["tf32"] = True
        if getattr(cfg, "MERGE_MAX_STEPS", None) is not None:
            ta_kwargs["max_steps"] = int(cfg.MERGE_MAX_STEPS)
        ta_kwargs[_eval_kw] = "no"

        args = make_train_args(**ta_kwargs)
        trainer = make_trainer(
            model=wrapper, args=args, tokenizer=tokenizer,
            train_dataset=chunked_train, eval_dataset=None,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[loss_cb],
        )
        try:
            if resume_ckpt:
                trainer.train(resume_from_checkpoint=resume_ckpt)
            else:
                trainer.train()
        finally:
            del trainer
            gc.collect()

        torch.save(wrapper.state_dict(), state_path)
        loss_cb.save_summary(summary_path)
        with open(os.path.join(out_dir, "train_log_history.json"), "w", encoding="utf-8") as hf:
            json.dump(loss_cb.history, hf, indent=2)
        pipe.mark(f"train_merger_{tag}_done -> {state_path}")
        return wrapper

    return pipe.run(f"train_merger_{tag}", _train)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(), "GPU must support bf16"
cfg.FP16 = False
cfg.BF16 = True
cfg.MERGE_BS = 64
cfg.MERGE_GRAD_ACCUM = 2
cfg.EVAL_BS = 32

pipe.mark("START ortho-only bf16 (resume if ckpt exists)")
merger_ortho_trained = train_merger(False, tag="ortho")
merger_ortho_trained = merger_ortho_trained.cpu()
_reclaim("after_ortho")
pipe.mark("ORTHO_DONE")
print(f"Done. Weights: {state_path_for('ortho')}", flush=True)


cfg.FP16 = False
cfg.BF16 = True
cfg.EVAL_BS = min(int(getattr(cfg, "EVAL_BS", 32)), 32)


def _drop(*names):
    g = globals()
    for n in names:
        if n in g and g[n] is not None:
            try:
                g[n].cpu()
            except Exception:
                pass
            g[n] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def coeff_grid():
    cs, c = [], float(cfg.COEFF_START)
    while c <= float(cfg.COEFF_END) + 1e-9:
        cs.append(float(round(c, 10)))
        c += float(cfg.COEFF_STEP)
    return cs


def make_eval_trainer(model):
    args = TrainingArguments(
        output_dir=os.path.join(cfg.OUT, "eval_tmp"),
        per_device_eval_batch_size=int(cfg.EVAL_BS),
        dataloader_drop_last=False, fp16=False, bf16=True, report_to="none",
        remove_unused_columns=False, no_cuda=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return make_trainer(model=model, args=args, tokenizer=tokenizer,
                        eval_dataset=chunked_test, data_collator=collator)


def barrier_from_curve(losses):
    cs = sorted(losses.keys())
    L0, L1 = losses[cs[0]], losses[cs[-1]]
    span = cs[-1] - cs[0]
    maxb = -1e9
    for c in cs:
        lam = (c - cs[0]) / span
        maxb = max(maxb, losses[c] - (lam * L1 + (1 - lam) * L0))
    return float(maxb)


def park_and_free(wrapper, tag):
    out_dir = os.path.join(cfg.OUT, f"merger_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    path = state_path_for(tag)
    torch.save(wrapper.state_dict(), path)
    try:
        wrapper.cpu()
    except Exception:
        pass
    del wrapper
    return path


def build_one(permutations_only, tag):
    w = build_merger(permutations_only, tag=tag)
    _fix_circuit_attn_meta(w)
    return park_and_free(w, tag)


def load_one(permutations_only, tag):
    path = state_path_for(tag)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    w = build_merger(permutations_only, tag=f"reload_{tag}")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    w.load_state_dict(sd)
    del sd
    _fix_circuit_attn_meta(w)
    return w.cpu()


@torch.no_grad()
def sweep_path(tag, permutations_only, name):
    print(f"=== {name} ({tag}) ===", flush=True)
    wrapper = load_one(permutations_only, tag).cuda().eval()
    trainer = make_eval_trainer(wrapper)
    losses = {}
    try:
        for c in coeff_grid():
            wrapper.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(c))
            trainer.model = wrapper
            losses[round(c, 4)] = float(trainer.evaluate()["eval_loss"])
            print(f"  lambda={c:.2f}  loss={losses[round(c,4)]:.4f}", flush=True)
    finally:
        try:
            wrapper.cpu()
        except Exception:
            pass
        del trainer, wrapper
    return losses


@torch.no_grad()
def sweep_vanilla():
    print("=== Vanilla averaging ===", flush=True)
    m = GPT2LMHeadModel.from_pretrained(merger_base_dirs[cfg.SEEDS[0]]).cuda().eval()
    sd_a = {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}
    m1 = GPT2LMHeadModel.from_pretrained(merger_base_dirs[cfg.SEEDS[1]])
    sd_b = {k: v.detach().cpu().contiguous() for k, v in m1.state_dict().items()}
    del m1
    gc.collect()
    trainer = make_eval_trainer(m)
    losses = {}
    try:
        for c in coeff_grid():
            blended = {}
            for k in sd_a:
                if torch.is_floating_point(sd_a[k]):
                    blended[k] = (c * sd_a[k] + (1.0 - c) * sd_b[k]).to("cuda")
                else:
                    blended[k] = sd_a[k]
            m.load_state_dict(blended, strict=False)
            del blended
            trainer.model = m
            losses[round(c, 4)] = float(trainer.evaluate()["eval_loss"])
            print(f"  lambda={c:.2f}  loss={losses[round(c,4)]:.4f}", flush=True)
    finally:
        del trainer, m, sd_a, sd_b
    return losses


_drop("merger_ortho_fresh", "merger_perm_fresh", "merger_ortho_trained", "merger_perm_trained")

jobs_build = [("ortho_fresh", False), ("perm_fresh", True)]
for tag, perm_only in jobs_build:
    if os.path.isfile(state_path_for(tag)):
        print(f"[skip build] {tag} already parked", flush=True)
    else:
        build_one(perm_only, tag)

assert os.path.isfile(state_path_for("ortho")), "Ortho merger training did not produce merger_state.pt"

curves = {}
curves["Vanilla averaging"] = sweep_vanilla()
curves["Weight matching (ours)"] = sweep_path("ortho_fresh", False, "WM ortho fresh")
curves["Activation matching"] = sweep_path("perm_fresh", True, "WM perm fresh")
curves["Learned matching (ours)"] = sweep_path("ortho", False, "learned ortho")

if os.path.isfile(state_path_for("perm")):
    curves["Learned matching (permutations)"] = sweep_path("perm", True, "learned perm")
else:
    print("merger_perm not found — run merge_permutations.py first for the 5th curve.", flush=True)

barriers = {k: barrier_from_curve(v) for k, v in curves.items()}
for k, b in barriers.items():
    print(f"{k:40s} barrier={b:.4f}", flush=True)


os.makedirs(cfg.OUT, exist_ok=True)
partial = "Learned matching (permutations)" not in curves

payload = {
    "curves": {k: {f"{a:.4f}": float(b) for a, b in sorted(v.items())} for k, v in curves.items()},
    "barriers": {k: float(v) for k, v in barriers.items()},
    "coeff_grid": sorted({float(a) for v in curves.values() for a in v.keys()}),
    "note": "partial table" if partial else "full table",
    "endpoints": {
        name: {
            "loss_lambda_min": float(curve[min(curve)]),
            "loss_lambda_max": float(curve[max(curve)]),
            "lambda_min": float(min(curve)),
            "lambda_max": float(max(curve)),
        }
        for name, curve in curves.items()
    },
}
json_path = os.path.join(cfg.OUT, "interp_curves_partial.json" if partial else "interp_curves.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print("Saved", json_path, flush=True)

import csv
csv_path = os.path.join(cfg.OUT, "interp_curves_partial.csv" if partial else "interp_curves.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method", "lambda", "eval_loss"])
    for name, curve in curves.items():
        for lam in sorted(curve):
            w.writerow([name, f"{lam:.4f}", f"{curve[lam]:.8f}"])

barrier_csv = os.path.join(cfg.OUT, "barriers_partial.csv" if partial else "barriers.csv")
with open(barrier_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method", "barrier"])
    for name, b in barriers.items():
        w.writerow([name, f"{b:.8f}"])

log_path = os.path.join(cfg.OUT, "eval_results_partial.log" if partial else "eval_results.log")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("GLMC BookCorpus lambda-sweep results\n")
    f.write(f"partial={partial}\n\n")
    for name, curve in curves.items():
        f.write(f"=== {name} ===\n")
        f.write(f"barrier={barriers[name]:.6f}\n")
        for lam in sorted(curve):
            f.write(f"  lambda={lam:.2f}  loss={curve[lam]:.6f}\n")
        f.write("\n")
    f.write("=== barrier summary ===\n")
    for name, b in barriers.items():
        f.write(f"{name:40s}  {b:.6f}\n")
print("Saved", log_path, flush=True)

style = {
    "Vanilla averaging": dict(color="tab:green", marker="D"),
    "Activation matching": dict(color="tab:purple", marker="s"),
    "Weight matching (ours)": dict(color="tab:orange", marker="v"),
    "Learned matching (permutations)": dict(color="tab:red", marker="^"),
    "Learned matching (ours)": dict(color="tab:blue", marker="o"),
}

fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
for name, curve in curves.items():
    xs = np.array(sorted(curve.keys()), dtype=float)
    ys = np.array([curve[x] for x in xs], dtype=float)
    st = style.get(name, dict(color="black", marker="o"))
    ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=5,
            markeredgecolor="black", linewidth=1.5, label=name)
ax.set_xlabel(r"Interpolation coefficient ($\lambda$)")
ax.set_ylabel("Loss")
ax.set_title("GLMC BookCorpus")
ax.grid(True, linestyle="dotted", alpha=0.5)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
png = os.path.join(cfg.OUT, "loss_interp_partial.png" if partial else "loss_interp_bookcorpus.png")
pdf = os.path.splitext(png)[0] + ".pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print("Saved", png, flush=True)
plt.close(fig)

if not partial:
    order = [
        "Vanilla averaging", "Activation matching", "Weight matching (ours)",
        "Learned matching (permutations)", "Learned matching (ours)",
    ]
    ranks = {n: r + 1 for r, n in enumerate(sorted(order, key=lambda k: barriers[k]))}
    rows = [{"Method": n, "BookCorpus (B)": f"{barriers[n]:.2f} ({ranks[n]})"} for n in order]
    with open(os.path.join(cfg.OUT, "results_table.json"), "w") as f:
        json.dump({n: {"barrier": barriers[n], "rank": ranks[n]} for n in order}, f, indent=2)
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))

print("\nBarriers:", flush=True)
for name, b in barriers.items():
    print(f"  {name:40s} {b:.4f}", flush=True)
