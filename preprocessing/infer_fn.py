"""vLLM inference function for LLM uncertainty estimation.

Runs batched LLM inference and captures per-token top-K logits for
downstream uncertainty quantification.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pyarrow.parquet as pq
import torch


@dataclass
class LLMConfig:
    """LLM inference configuration."""
    llm_name: str = ""
    log_dir: str = "log/"
    device: str = "cuda"
    max_model_len: int = None
    max_input_len: int = 32 * 1024
    output_token_len: int = 8 * 1024
    temperature: float = 1.0
    top_k: int = 20
    pe_method: str = "raw"
    eager_mode: bool = True


def _logprobs_to_topk_list(
    logprobs_list: Optional[List[Dict[int, Any]]],
    num_topk: int = 20,
) -> Optional[List[List[Dict[str, Any]]]]:
    """Convert vLLM logprob dicts into a list of top-k entries per token."""
    if logprobs_list is None:
        return None
    topk_all: List[List[Dict[str, Any]]] = []
    for cand_dict in logprobs_list:
        if cand_dict is None:
            topk_all.append(None)
            continue
        entries: List[Dict[str, Any]] = []
        chosen_entry: Optional[Dict[str, Any]] = None
        for tid, v in cand_dict.items():
            ent = {
                "token_id": tid,
                "token": getattr(v, "decoded_token", None),
                "logit": float(getattr(v, "logprob", 0.0)),
                "rank": getattr(v, "rank", None),
            }
            entries.append(ent)
            if ent["rank"] == 0:
                chosen_entry = ent
        if all(e["rank"] is not None for e in entries):
            entries_sorted = sorted(entries, key=lambda e: e["rank"])
        else:
            entries_sorted = sorted(entries, key=lambda e: e["logit"], reverse=True)
        topk = entries_sorted[:num_topk]
        if chosen_entry is not None and all(t["token_id"] != chosen_entry["token_id"] for t in topk):
            topk.append(chosen_entry)
        topk_all.append(topk)
    return topk_all


@torch.no_grad()
def infer_llm_on_dataset(
    llm,
    dataset,
    sampling_params,
    algo_config: LLMConfig,
    task_type: str = "qa",
    n_samples: int = None,
    output_fpath: Optional[str] = None,
    logits_out_fpath: Optional[str] = None,
):
    """Run vLLM inference on dataset and capture top-K logits per token.

    Streaming mode: if output_fpath and logits_out_fpath are provided, writes
    results/logits incrementally without large in-memory DataFrames.

    Returns:
        (None, None) in streaming mode.
        (results_df, logits_df) in non-streaming mode.
    """
    if n_samples is None:
        prompts = dataset["prompt"]
        labels = dataset["label"]
    else:
        n_samples = min(n_samples, len(dataset["prompt"]))
        prompts = dataset["prompt"][:n_samples]
        labels = dataset["label"][:n_samples]

    responses = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    logits_rows: List[Dict[str, Any]] = [{} for _ in responses]
    num_outputs = len(responses[0].outputs) if responses else 0
    results: Dict[str, Any] = {"prompt": prompts, "label": labels}
    for i_output in range(num_outputs):
        results.update({
            f"generated_text_{i_output}": [],
            f"answer_{i_output}": [],
            f"answer_token_index_{i_output}": [],
            f"answer_token_topk_logits_{i_output}": [],
        })

    for r_idx, response in enumerate(responses):
        for i_output in range(num_outputs):
            output = response.outputs[i_output]
            gen_topk = _logprobs_to_topk_list(output.logprobs)
            logits_rows[r_idx][f"gen_topk_logits_{i_output}"] = gen_topk

            results[f"generated_text_{i_output}"].append(output.text)
            # For QA: full response text is the answer
            results[f"answer_{i_output}"].append(output.text.strip())
            last_token_idx = len(gen_topk) - 1 if gen_topk else None
            last_token_topk = gen_topk[-1] if gen_topk else None
            results[f"answer_token_index_{i_output}"].append(last_token_idx)
            results[f"answer_token_topk_logits_{i_output}"].append(
                [last_token_topk] if last_token_topk is not None else None
            )

    results_df = pl.DataFrame(results, strict=False)
    logits_df = pl.DataFrame(logits_rows, strict=False)

    if output_fpath is not None and logits_out_fpath is not None:
        pq.write_table(results_df.to_arrow(), str(output_fpath))
        pq.write_table(logits_df.to_arrow(), str(logits_out_fpath))
        return None, None

    return results_df, logits_df
