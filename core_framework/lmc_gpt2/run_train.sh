python train.py --seed 0 --epochs 100 --early_stop --early_stop_patience 5 --splits_dir splits_tiny_contig --wandb --wandb_project gpt2-merging-demo --wandb_group tinyshakes --n_embd 256

python train.py --seed 1 --epochs 100 --early_stop --early_stop_patience 5 --splits_dir splits_tiny_contig --wandb --wandb_project gpt2-merging-demo --wandb_group tinyshakes --n_embd 256

python train.py --seed 2 --epochs 100 --early_stop --early_stop_patience 5 --splits_dir splits_tiny_contig --wandb --wandb_project gpt2-merging-demo --wandb_group tinyshakes --n_embd 512