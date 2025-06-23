#!/bin/bash

# Activate Conda environment
# For macOS (commented out):
# source /Users/chinonsoosuji/opt/anaconda3/etc/profile.d/conda.sh

# For Ubuntu/Linux:
source /home/chinonso/anaconda3/etc/profile.d/conda.sh
conda activate lang2

# Configuration
XML_PATH="data/long-inputs_GREC_3_10.xml"
OUTPUT_PATH="result/output.json"
SUPPLIER="openai"
TASK_TYPE="translation"  # "generation" or "translation"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Run Python prediction script
python generate.py \
  --model_provider "$SUPPLIER" \
  --output_path "$OUTPUT_PATH" \
  --xml_path "$XML_PATH" \
  --task_type "$TASK_TYPE"

# Usage:
# chmod +x run.sh
# ./run.sh
