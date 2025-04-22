#! /bin/bash
set -euo pipefail

MODEL="model_hub/Qwen1.5-1.8B/WSPIN/ite0"
INPUT="data/Ultrachat200k/WSPIN/ite0/train.jsonl"
OUTDIR="data/Ultrachat200k/WSPIN/ite1"
BATCH=8
MAX_NEW=256
FRAC_LEN=1000000

# loop over data_frac = 0..70
for frac in {0..1}; do
  python generate.py \
    --model       "$MODEL" \
    --input_dir   "$INPUT" \
    --output_dir  "$OUTDIR" \
    --batch_size  $BATCH \
    --max_new_tokens $MAX_NEW \
    --data_frac   $frac \
    --frac_len    $FRAC_LEN \
    --split       train
done
