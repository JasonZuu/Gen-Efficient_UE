#!/bin/bash
# Step 1: Run CoQA LLM inference with Gemma-4-E4B
# Generates result and logits parquets for train/val/test splits.

MODEL_PATH="google/gemma-4-e4b-it"
OUTPUT_DIR="data/llm-inference"
GPU_IDS="0"
NUM_RESPONSES=1

for SPLIT in train val test; do
    echo "=== Running inference for split: $SPLIT ==="
    python preprocessing/infer_llm.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --data_split "$SPLIT" \
        --num_responses "$NUM_RESPONSES" \
        --gpu_ids "$GPU_IDS"
done

echo "=== Preprocessing complete ==="
echo "Results saved to: $OUTPUT_DIR/coqa/"
