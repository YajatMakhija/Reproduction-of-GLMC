
import os
import sys
import subprocess

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.50.1", "datasets>=2.14,<2.20", "einops",
    "safetensors", "accelerate>=0.26.0", "scipy", "matplotlib", "pandas",
    "huggingface_hub>=0.23.0",
], check=False)

import json
import time
import gc
import re
from itertools import chain

import numpy as np
import torch

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    set_seed,
)
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required for training.")
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

    EVAL_BS = 64
    BF16 = False
    FP16 = True
    SEEDS = (0, 1)
    TOKENIZER = "gpt2"
    OUT = "./glmc_bookcorpus_out"
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
    cfg.TRAIN_BS = 4
    cfg.EVAL_BS = 4
    cfg.GRAD_ACCUM = 1
    cfg.MAX_STEPS = 6
    cfg.SAVE_STEPS = 3
    cfg.NUM_PROC = 1
    cfg.SENTS_PER_DOC = 8
    print(f"*** SMOKE MODE *** out={cfg.OUT}")
else:
    print(
        f"PAPER-LIKE | BookCorpus | Block={cfg.BLOCK_SIZE} | embd={cfg.N_EMBD} | "
        f"train_epochs={cfg.TRAIN_EPOCHS} | train_bs={cfg.TRAIN_BS} | "
        f"fp16={cfg.FP16} bf16={cfg.BF16} | out={cfg.OUT}"
    )


def _reclaim(msg=""):
    gc.collect()
    if msg:
        print(f"  [ram] {msg}", flush=True)


cache_tag = "smoke_v1" if cfg.SMOKE else "paper_v2_full"
cache_root = os.path.join(cfg.OUT, f"data_cache_{cache_tag}")
chunked_dir = os.path.join(cache_root, "chunked")
os.makedirs(cache_root, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = int(10 ** 9)

nproc = 1
pack_bs = int(getattr(cfg, "DATA_PACK_BATCH", 400))
tok_bs = int(getattr(cfg, "DATA_TOK_BATCH", 32))
chunk_bs = int(getattr(cfg, "DATA_CHUNK_BATCH", 16))
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

    import shutil
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

print(
    f"READY {'SMOKE' if cfg.SMOKE else 'PAPER'} | blocks train/val/test: "
    f"{len(chunked_train):,} / {len(chunked_val):,} / {len(chunked_test):,}",
    flush=True,
)


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
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.history = []
        open(self.jsonl_path, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": int(state.global_step),
               "epoch": float(state.epoch) if state.epoch is not None else None}
        for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
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
            json.dump({"history": self.history}, f, indent=2)


def build_model(seed):
    set_seed(seed)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.BLOCK_SIZE,
        n_ctx=cfg.BLOCK_SIZE,
        n_embd=cfg.N_EMBD,
        n_layer=cfg.N_LAYER,
        n_head=cfg.N_HEAD,
        n_inner=cfg.N_INNER,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return GPT2LMHeadModel(config)


def train_seed(seed):
    out_dir = os.path.join(cfg.OUT, f"gpt2_bookcorpus_seed{seed}_nembd{cfg.N_EMBD}")
    logs_dir = os.path.join(cfg.OUT, "train_logs")
    os.makedirs(logs_dir, exist_ok=True)
    jsonl_path = os.path.join(logs_dir, f"seed{seed}_train.jsonl")
    summary_path = os.path.join(logs_dir, f"seed{seed}_train.json")

    if os.path.isfile(os.path.join(out_dir, "config.json")) and \
       os.path.isfile(os.path.join(out_dir, "training_complete.flag")):
        print(f"[skip] seed {seed} already trained: {out_dir}", flush=True)
        return out_dir

    resume_ckpt = find_latest_checkpoint(out_dir)
    if resume_ckpt:
        print(f"*** seed {seed}: RESUME from {resume_ckpt} ***", flush=True)
        model = GPT2LMHeadModel.from_pretrained(resume_ckpt)
    else:
        print(f"seed {seed}: training from scratch -> {out_dir}", flush=True)
        model = build_model(seed)

    loss_cb = LossLoggerCallback(jsonl_path)
    ta_kwargs = dict(
        output_dir=out_dir,
        logging_strategy="steps", logging_steps=1 if cfg.SMOKE else 50,
        save_strategy="steps", save_steps=int(cfg.SAVE_STEPS),
        save_total_limit=int(cfg.SAVE_TOTAL_LIMIT),
        num_train_epochs=float(cfg.TRAIN_EPOCHS),
        per_device_train_batch_size=int(cfg.TRAIN_BS),
        per_device_eval_batch_size=int(cfg.EVAL_BS),
        gradient_accumulation_steps=int(cfg.GRAD_ACCUM),
        learning_rate=cfg.LR, lr_scheduler_type="cosine",
        warmup_ratio=cfg.WARMUP_RATIO, weight_decay=cfg.WEIGHT_DECAY,
        fp16=cfg.FP16, bf16=cfg.BF16, report_to="none",
        dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, no_cuda=False,
        eval_strategy="steps", eval_steps=int(cfg.SAVE_STEPS),
        seed=seed, data_seed=seed,
    )
    if cfg.MAX_STEPS is not None:
        ta_kwargs["max_steps"] = int(cfg.MAX_STEPS)
    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model, args=args, tokenizer=tokenizer,
        train_dataset=chunked_train, eval_dataset=chunked_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[loss_cb],
    )
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    loss_cb.save_summary(summary_path)
    with open(os.path.join(out_dir, "training_complete.flag"), "w") as f:
        f.write("done\n")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"seed {seed}: done -> {out_dir}", flush=True)
    return out_dir


for seed in cfg.SEEDS:
    train_seed(seed)

print("Both base models trained. Run merge_and_evaluate.py next "
      "(and merge_permutations.py if you want the 5th curve).", flush=True)
