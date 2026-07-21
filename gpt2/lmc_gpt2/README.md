# Generalized Linear Mode Connectivity for Transformers

Official repository for the NeurIPS 2025 paper  
**[Generalized Linear Mode Connectivity for Transformers](https://openreview.net/pdf/74f91986c42f68d256ffcef1d4a4ee46f7530d3d.pdf)**.

Currently includes the code for the **Tiny Shakespeare experiments**.  
Support for multi-model merging and additional datasets will be added in future releases.


## Installation

Create and activate a Python environment (e.g. Python 3.11), then run:

```bash
pip install -r requirements.txt
```
------------------------------------------------------------------------

## 1. Create train/val/test splits

``` bash
python create_splits.py --data_file tinyshakespeare.txt --out_dir ./splits_tiny_contig --val_frac 0.05 --test_frac 0.05 --method contiguous
```

------------------------------------------------------------------------

## 2. Compute token frequencies

``` bash
python compute_token_freqs.py --splits_dir ./splits_tiny_contig --tokenizer_dir ./gpt2_tokenizer --out_name token_freqs.pt
```

(Useful for weight matching: tokens never seen in training can be ignored.)

------------------------------------------------------------------------

## 3. Train multiple GPT-2 models
Trains three independent models. One of them has a larger embedding dimension.
``` bash
bash run_train.sh
```

------------------------------------------------------------------------

## 4. Merge models
Merges independently trained models under different settings.
1. Using our orthogonal alignment strategies.
2. Only using strict permutations.
3. Merging models of different embedding dimensions.
``` bash
bash run_merge.sh
```

------------------------------------------------------------------------

## 5. Evaluate + plot
Evaluates and plots the results.
``` bash
bash run_eval.sh
```

------------------------------------------------------------------------

## Citation
```
@inproceedings{theus2025glmc,
  title     = {Generalized Linear Mode Connectivity for Transformers},
  author    = {Theus, Alexander and Cabodi, Alessandro and Anagnostidis, Sotiris and Orvieto, Antonio and Singh, Sidak Pal and Boeva, Valentina},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

