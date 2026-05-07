#!/bin/bash
# Step 2: Evaluate Logit Magnitude UQ on CoQA test split.

INFERENCE_DIR="data/llm-inference/coqa"
MODEL_TAG="gemma-4-e4b-it"     # matches Path(model_path).name from preprocessing step
SPLIT="test"
MAX_INPUT_LEN="32k"
N_RESPONSES=1
M=5    # min-heap size for patience aggregation
W=20   # patience window (consecutive non-updates to halt)

RESULT_PARQUET="${INFERENCE_DIR}/coqa_${MAX_INPUT_LEN}_${MODEL_TAG}_raw_nres-${N_RESPONSES}_set-${SPLIT}.parquet"
LOGITS_PARQUET="${INFERENCE_DIR}/coqa_${MAX_INPUT_LEN}_${MODEL_TAG}_raw_nres-${N_RESPONSES}_logits_set-${SPLIT}.parquet"

echo "=== Evaluating Logit Magnitude UQ ==="
python evaluate_logit_magnitude.py \
    --result_parquet "$RESULT_PARQUET" \
    --logits_parquet "$LOGITS_PARQUET" \
    --dataset coqa \
    --M "$M" \
    --W "$W"
