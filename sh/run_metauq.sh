#!/bin/bash
# Step 3: Full MetaUE pipeline — encode embeddings, train via sweep, evaluate.

INFERENCE_DIR="data/llm-inference"
EMB_DIR="data/metaue_embeddings"
LOG_DIR="data/metaue_sweep"
DATASET="coqa"
MODEL_TAG="gemma-4-e4b-it"     # matches Path(model_path).name from preprocessing step
MAX_INPUT_LEN="32k"
N_RESPONSES=1
GPU_IDS="0"
SPLIT="test"

RESULT_PARQUET="${INFERENCE_DIR}/coqa/coqa_${MAX_INPUT_LEN}_${MODEL_TAG}_raw_nres-${N_RESPONSES}_set-${SPLIT}.parquet"
LOGITS_PARQUET="${INFERENCE_DIR}/coqa/coqa_${MAX_INPUT_LEN}_${MODEL_TAG}_raw_nres-${N_RESPONSES}_logits_set-${SPLIT}.parquet"
EMB_PARQUET="${EMB_DIR}/${DATASET}_Qwen3-VL-Embedding-2B/${SPLIT}.parquet"

# --- Step 3-i: Encode prompts to embeddings (run once per dataset) ---
echo "=== Encoding prompts to embeddings ==="
python encode_metaue.py \
    --dataset "$DATASET" \
    --inference_dir "$INFERENCE_DIR" \
    --output_dir "$EMB_DIR" \
    --splits train val test \
    --gpu_ids "$GPU_IDS"

# --- Step 3-ii: Run WandB sweep training (offline mode) ---
echo "=== Running MetaUE sweep training ==="
python sweep_metaue.py \
    --dataset "$DATASET" \
    --inference_dir "$INFERENCE_DIR" \
    --log_dir "$LOG_DIR" \
    --gpu_ids "$GPU_IDS"

# --- Step 3-iii: Evaluate the best trained model (path written by sweep_metaue.py) ---
BEST_MODEL=$(cat "${LOG_DIR}/last_best_model_path.txt")
echo "=== Evaluating MetaUE model: ${BEST_MODEL} ==="
python evaluate_metaue.py \
    --result_parquet "$RESULT_PARQUET" \
    --logits_parquet "$LOGITS_PARQUET" \
    --embedding_parquet "$EMB_PARQUET" \
    --model_path "$BEST_MODEL" \
    --dataset "$DATASET"
