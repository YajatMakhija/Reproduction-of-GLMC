# Generalized Linear Mode Connectivity for ViTs (CIFAR-10)

Two independently trained ViTs sit in different loss basins, so averaging their
weights gives a broken model (a large **loss barrier**). This repo aligns them
so they can be merged:

- **Weight matching (WM)** — data-free alignment (one orthogonal rotation of the
  residual stream + per-layer permutations of attention heads and MLP neurons).
- **Learned matching (LM)** — train the alignment to minimize the interpolated
  model's loss, driving the barrier to ~0 (the merged model matches its parents).

The model is a standard LayerNorm ViT with biases (mean pooling, QKV bias-free).
Internally it is reparameterized to a parameter-free-RMSNorm form on a centered
residual stream (`reparameterize`), which is what makes the alignment valid.

## Install

```bash
pip install -r requirements.txt
```

## Run

End to end (train 2 ViTs, weight-match + learned-match, report barriers):

```bash
python run_pipeline.py --config config.yaml            # GPU
python run_pipeline.py --smoke --device cpu            # fast correctness check
```

Or step by step:

```bash
python train_model.py  --config config.yaml --seed 0
python train_model.py  --config config.yaml --seed 1
python train_merger.py --folder0 outputs/models/run_A --folder1 outputs/models/run_B
python eval_barrier.py --folder0 outputs/models/run_A --folder1 outputs/models/run_B \
                       --merger-dir outputs/mergers/run_C
python tests/test_pipeline.py                          # function-preservation checks
```

## Barrier

For interpolation coefficient `λ ∈ {0.0, …, 1.0}` (`λ=1`→model 0, `λ=0`→model 1):
`B = maxλ [ L(λ·θ0 + (1-λ)·θ1) − (λ·L(θ0) + (1-λ)·L(θ1)) ]`. Expect
`VAN ≫ WM` and `LM ≈ 0`.

## Files

| file | role |
|------|------|
| `models/vit.py`      | `ViT` (LayerNorm+biases) and `ReparamViT` (RMSNorm form) + `reparameterize` |
| `weight_matching.py` | data-free alignment (SVD Procrustes + optimal-transport permutations) |
| `merger.py`          | `ViTMerger` — differentiable interpolation; only the alignment matrices train |
| `utils.py`, `enums.py` | interpolation, manifold projection, attention-circuit form |
| `barrier.py`         | loss-barrier metric + interpolation sweeps |
| `train_model.py`, `train_merger.py`, `eval_barrier.py`, `run_pipeline.py` | scripts |
