#!/bin/bash

#SBATCH --gres=gpu:a100:1
# SBATCH --nodelist=g129
#SBATCH -p compute
#SBATCH -J GEN
#SBATCH -t 1-23:59:59

# Activate Conda environment
# For macOS (commented out):
# source /Users/chinonsoosuji/opt/anaconda3/etc/profile.d/conda.sh

# For Ubuntu/Linux:
# source /home/chinonso/anaconda3/etc/profile.d/conda.sh
# conda activate lang2

# For ADAPT Cluster
source /home/cosuji/anaconda3/etc/profile.d/conda.sh
conda activate hf311

# Configuration
pilot="data/pilot.xml"
dev="data/dev.xml"
test="data/test.xml"
XML_PATH="$pilot"  # Change this to $pilot, $dev, or $test
SUPPLIER="hf"
MODEL="llama_en_lora"
TASK_TYPE="generation"  # "generation", "irish_generation", or "translation"

# Automatically infer dataset name
DATASET_NAME=$(basename "$XML_PATH" .xml)

# Customize output filename
if [ "$TASK_TYPE" = "translation" ]; then
  OUTPUT_PATH="results/${DATASET_NAME}_${MODEL}_translated.json"
elif [ "$TASK_TYPE" = "irish_generation" ]; then
  OUTPUT_PATH="results/${DATASET_NAME}_${MODEL}_GA.json"
else
  OUTPUT_PATH="results/${DATASET_NAME}_${MODEL}_EN.json"
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Construct base command
CMD="python generate.py \
  --model_provider \"$SUPPLIER\" \
  --model \"$MODEL\" \
  --output_path \"$OUTPUT_PATH\" \
  --xml_path \"$XML_PATH\" \
  --task_type \"$TASK_TYPE\""

# If translating, add translate_input argument
if [ "$TASK_TYPE" = "translation" ]; then
  CMD="$CMD --translate_input \"$OUTPUT_PATH\""
fi

# Execute command
eval $CMD

# Usage:
# chmod +x run.sh
# ./run.sh
# models:
# groq: llama3-70b-8192, llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, qwen/qwen3-32b
# openai o3 ($2/$8), gpt-4.1 ($2/$8)
# anthropic: claude-3-5-sonnet-latest ($3/$15), claude-3-7-sonnet-latest ($3/$15) 
# claude-3-5-haiku-latest ($0.8/$4), claude-3-haiku-latest ($0.25/$1.25)