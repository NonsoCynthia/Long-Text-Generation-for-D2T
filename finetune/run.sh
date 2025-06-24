#!/bin/bash

#SBATCH --gres=gpu:rtxa6000:1
#SBATCH -p compute
#SBATCH -J UCCIX
#SBATCH -t 1-23:59:59

source /home/cosuji/anaconda3/etc/profile.d/conda.sh
conda activate hugface

# 1 – Extract triples & Irish sentences
# autoflake -i ../data/ga_train.xml  # optional: ensure file encoding is UTF‑8
# python finetune.py extract \
#        --input ../data/ga_train.xml \
#        --output ../data/ga_train_parsed.jsonl

# 2 – Fine‑tune with LoRA (needs ≈ 1×A100‑80GB or 2×40GB GPUs)
python finetune.py finetune \
       --train ../data/ga_train_parsed.jsonl \
       --outdir ./uccix_ga_lora

# # 3 – Generate Irish text
# python finetune.py generate \
#        --model ./uccix_ga_lora \
#        --triples "Dublin|located_in|Ireland|population|1200000"
