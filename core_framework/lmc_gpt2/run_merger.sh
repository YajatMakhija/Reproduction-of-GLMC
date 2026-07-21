MODEL0="./gpt2_tinyshakespeare_seed0_nembd256"
MODEL1="./gpt2_tinyshakespeare_seed1_nembd256"

# Merge with orthogonal alignment
python train_merger.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir merge_seed_0_1 \
  --epochs 10 \
  --batch_size 32 \
  --fp16 \
  --splits_dir splits_tiny_contig \
  --early_stop \
  --sampler NARROW_UNIFORM \
  --wandb --wandb_project gpt2-merging-demo --wandb_group tinyshakes-merge


# Merge with permutation alignment (we now pass --permutations_only )
python train_merger.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir merge_seed_0_1_perm \
  --epochs 10 \
  --batch_size 32 \
  --permutations_only \
  --fp16 \
  --splits_dir splits_tiny_contig \
  --early_stop \
  --sampler NARROW_UNIFORM \
  --wandb --wandb_project gpt2-merging-demo --wandb_group tinyshakes-merge


MODEL0="./gpt2_tinyshakespeare_seed2_nembd512"
MODEL1="./gpt2_tinyshakespeare_seed0_nembd256"

# Merge models with heterogeneous embedding dimensions
python train_merger.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir merge_seed_2_0_width_hetero \
  --epochs 10 \
  --batch_size 32 \
  --fp16 \
  --splits_dir splits_tiny_contig \
  --early_stop \
  --sampler NARROW_UNIFORM_BIASED \
  --wandb --wandb_project gpt2-merging-demo --wandb_group tinyshakes-merge

