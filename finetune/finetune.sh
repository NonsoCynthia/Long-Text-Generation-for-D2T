#!/bin/bash

#SBATCH --gres=gpu:a100:1
# SBATCH --nodelist=g129
#SBATCH -p long
#SBATCH -J FT
#SBATCH -t 9-23:59:59

source /home/cosuji/anaconda3/etc/profile.d/conda.sh
conda activate hugface

# set -e  # Optional: stop if any command fails

# Set variables (edit these as needed)
LANG="ga"   # en or ga
OUTDIR="webnlg_llama2_13b_en"
TRAIN_FILE="../data/ga_train.jsonl"
DEV_FILE="../data/ga_dev.jsonl"

# # ---- ENGLISH: GEM/web_nlg, LoRA ----
# python finetune_model.py \
#     --lang en \
#     --peft_type lora \
#     --outdir "$OUTDIR"

# ---- IRISH: Local JSONL, Prompt Tuning ----
python finetune_model.py \
    --lang ga \
    --peft_type prompt \
    --num_virtual_tokens 16 \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --outdir "webnlg_llama2_13b_ga_prompt"

# # ---- IRISH: Prefix Tuning ----
# python finetune_model.py \
#     --lang ga \
#     --peft_type prefix \
#     --num_virtual_tokens 20 \
#     --prefix_projection \
#     --train_file "$TRAIN_FILE" \
#     --dev_file "$DEV_FILE" \
#     --outdir "webnlg_llama2_13b_ga_prefix"
