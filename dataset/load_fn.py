"""CoQA dataset loader using HuggingFace datasets."""

from datasets import load_dataset
from dataset.map_fn import coqa_map_batch_fn
from pathlib import Path
import yaml


def _load_prompt_config():
    prompt_path = Path(__file__).resolve().parent / "prompts" / "coqa.yaml"
    with open(prompt_path, "r") as f:
        return yaml.safe_load(f)


def load_coqa(data_split: str = "train", pe_method: str = "raw"):
    """Load and format CoQA dataset for LLM inference.

    Downloads CoQA from HuggingFace Hub. The original dataset has only train
    and validation splits; we use 80/10/10 of train for train/val, and the
    HF validation split as test.

    Args:
        data_split: "train", "val", or "test".
        pe_method:  "raw" or "cot" (chain-of-thought suffix).

    Returns:
        HuggingFace Dataset with columns: prompt, label, data_source, ground_truth.
    """
    instruction = " Let's think step by step." if pe_method == "cot" else ""
    prompt_config = _load_prompt_config()

    raw = load_dataset("coqa", trust_remote_code=True)

    if data_split == "test":
        hf_split = raw["validation"]
    else:
        full_train = raw["train"]
        n = len(full_train)
        n_val = n // 10
        n_train = n - 2 * n_val
        if data_split == "train":
            hf_split = full_train.select(range(n_train))
        else:  # val
            hf_split = full_train.select(range(n_train, n_train + n_val))

    dataset = hf_split.map(
        coqa_map_batch_fn,
        batched=True,
        fn_kwargs={
            "instruction": instruction,
            "llm_tokenizer": None,
            "max_input_length": 32 * 1024,
            "system_prompt": None,
            "prompt_config": prompt_config,
        },
        remove_columns=["source", "story", "questions", "answers"],
    )
    return dataset
