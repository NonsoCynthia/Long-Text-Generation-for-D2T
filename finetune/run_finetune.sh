#!/bin/bash

#SBATCH --gres=gpu:a100:1
# SBATCH --nodelist=g129
#SBATCH -p compute
#SBATCH -J FT
#SBATCH -t 1-23:59:59

source /home/cosuji/anaconda3/etc/profile.d/conda.sh
conda activate hugface

LANG="ga"   # en or ga
XML=../data/ga_train.xml
JSON=../data/${LANG}_train_parsed.jsonl
OUTDIR=llama_${LANG}_lora

# # 1 – extract
# python finetune_extract_triples.py --lang $LANG \
#                                    --input "$XML" \
#                                    --output "$JSON"

# 2 – fine-tune
python finetune_model.py --lang $LANG \
                         --train "$JSON" \
                         --outdir "$OUTDIR"

# # 3 – smoke-test generation
# python finetune_generate.py --lang $LANG \
#                             --model "$OUTDIR" \
#                             --triples "London|located_in|England|population|8,799,800"
