# Generation-Efficient Uncertainty Quantification for LLMs

## Methods

### Logit Magnitude
Per-token uncertainty is measured as the L2 norm of the top-M logits at each generated token position. Token-level scores are aggregated with a **patience rule**: a min-heap tracks the M largest scores seen so far; scanning halts when W consecutive tokens fail to update the heap. The final uncertainty is the mean of the M heap values.

### MetaUE
A small 2-layer MLP trained on **frozen prompt embeddings** (Qwen3-VL-Embedding-2B) to predict logit_magnitude scores. Because it is conditioned only on the prompt (not the model output), MetaUE is cheap to compute at inference time. Training uses MSE loss on raw logit_magnitude scores.

---

## Requirements

```bash
pip install torch transformers vllm polars numpy scikit-learn rouge_score wandb tqdm datasets pyyaml
```

GPU with ≥ 24 GB VRAM recommended for Gemma-4-E4B inference and Qwen3-VL-Embedding encoding.

---

## Pipeline Overview

```
Step 1  Preprocessing  →  inference parquets (result + logits)
Step 2  Logit Magnitude →  AUROC / AURAC / Balanced Accuracy metrics
Step 3  MetaUE         →  encode embeddings → sweep train → evaluate
```

All commands below should be run from this directory (`Logit Magnitude and MetaUE/`).

---

## Step 1 — Preprocessing (CoQA + Gemma-4-E4B)

Run vLLM inference for all three splits. Each call saves a result parquet and a logits parquet to `data/llm-inference/coqa/`.

**Python (one split at a time):**

```bash
# Train split
python preprocessing/infer_llm.py \
    --model_path google/gemma-4-e4b-it \
    --output_dir data/llm-inference \
    --data_split train \
    --num_responses 1 \
    --gpu_ids 0

# Validation split
python preprocessing/infer_llm.py \
    --model_path google/gemma-4-e4b-it \
    --output_dir data/llm-inference \
    --data_split val \
    --num_responses 1 \
    --gpu_ids 0

# Test split
python preprocessing/infer_llm.py \
    --model_path google/gemma-4-e4b-it \
    --output_dir data/llm-inference \
    --data_split test \
    --num_responses 1 \
    --gpu_ids 0
```

**Or run all splits with the shell script:**

```bash
bash sh/preprocessing.sh
```

Output files follow the naming pattern:
```
data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_set-{split}.parquet
data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_logits_set-{split}.parquet
```

---

## Step 2 — Logit Magnitude

Evaluate Logit Magnitude UQ directly from the inference parquets (no training needed).

**Python:**

```bash
python evaluate_logit_magnitude.py \
    --result_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_set-test.parquet \
    --logits_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_logits_set-test.parquet \
    --dataset coqa \
    --M 5 \
    --W 20
```

Key arguments:
- `--M` — min-heap size (number of top scores to track)
- `--W` — patience window (consecutive non-updates before halting)
- `--output_path` — optional path to save metrics as JSON

**Or run with the shell script:**

```bash
bash sh/run_logit_magnitude.sh
```

---

## Step 3 — MetaUE

MetaUE requires three sub-steps: encoding, training, and evaluation.

### 3-i — Encode prompts to embeddings

Run once per dataset. Embeddings are independent of which LLM produced the inference data.

```bash
python encode_metaue.py \
    --dataset coqa \
    --inference_dir data/llm-inference \
    --output_dir data/metaue_embeddings \
    --splits train val test \
    --gpu_ids 0
```

Output: `data/metaue_embeddings/coqa_Qwen3-VL-Embedding-2B/{train,val,test}.parquet`

### 3-ii — Train MetaUE (WandB sweep, offline)

Runs a grid search over learning rate, dropout, and batch size. WandB runs offline — no account required.

```bash
python sweep_metaue.py \
    --dataset coqa \
    --inference_dir data/llm-inference \
    --log_dir data/metaue_sweep \
    --gpu_ids 0
```

After the sweep, the script re-trains with the best hyperparameters over 5 seeds, saves the best model to `data/metaue_sweep/.../trained/best_model.pt`, and writes its path to `data/metaue_sweep/last_best_model_path.txt`.

### 3-iii — Evaluate trained MetaUE model

```bash
BEST_MODEL=$(cat data/metaue_sweep/last_best_model_path.txt)
python evaluate_metaue.py \
    --result_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_set-test.parquet \
    --logits_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_logits_set-test.parquet \
    --embedding_parquet data/metaue_embeddings/coqa_Qwen3-VL-Embedding-2B/test.parquet \
    --model_path "$BEST_MODEL" \
    --dataset coqa
```

**Or run all three sub-steps with the shell script:**

```bash
bash sh/run_metaue.sh
```

---

## Data Format

### Result parquet (columns per row = one dataset sample)

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full formatted prompt sent to the LLM |
| `label` | str | Ground-truth answer |
| `answer_0` | str | LLM-generated response |

### Logits parquet (one row per sample)

| Column | Type | Description |
|--------|------|-------------|
| `gen_topk_logits_0` | list[list[dict]] | Per-token top-M `{token_id: logit}` pairs for response 0 |

### Embedding parquet

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Prompt text |
| `label` | str | Ground-truth answer |
| `embedding` | list[float] | Prompt embedding vector |

---

## Output Metrics

| Metric | Description |
|--------|-------------|
| AUROC | Area under the ROC curve (uncertainty → error detection) |
| AURAC | Area under the risk-acceptance curve (fraction accepted vs. accuracy) |
| Balanced Accuracy | Mean of sensitivity and specificity at the optimal Youden threshold |

Higher AUROC/AURAC indicates better uncertainty ordering.
Higher Balanced Accuracy indicates better binary error detection at the optimal threshold.
