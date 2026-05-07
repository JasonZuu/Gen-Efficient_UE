"""WandB sweep for MetaUE hyperparameter search (MLP-only, embeddings pre-computed).

Trains MetaUE with logit_magnitude labels using MSE loss. Runs a grid search
over lr, dropout, batch_size, then trains 5 seeds with the best hyperparams.

Pre-requisite: run encode_metaue.py first to produce embedding parquets.

Usage
-----
python sweep_metaue.py \\
    --dataset coqa \\
    --model_path google/gemma-4-e4b-it \\
    --inference_dir data/llm-inference \\
    --project metaue-demo \\
    --gpu_ids 0
"""

import argparse
import os
import sys

# Set CUDA_VISIBLE_DEVICES before any CUDA-aware import
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu_ids", type=lambda s: [int(x) for x in s.split(",")], default=[0])
_pre_args, _ = _pre_parser.parse_known_args(sys.argv[1:])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, _pre_args.gpu_ids))
os.environ["WANDB_MODE"] = "offline"

import gc
import json
import random
import resource
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl
import torch
import wandb

if torch.cuda.is_available():
    _gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
          f"→ {torch.cuda.device_count()} visible device(s): {_gpu_names}")
    torch.cuda.set_device(0)
else:
    print("[GPU] No CUDA devices visible — running on CPU")

from metaue_src.config import DEFAULT_CONFIG, _model_tag_from_path
from metaue_src.dataset import MetaUEDataset
from metaue_src.encode import load_embedding_parquet, embedding_parquet_path
from metaue_src.evaluate import evaluate_metaue
from metaue_src.label_functions import (
    LABEL_FN_IS_REGRESSION,
    LABEL_FN_NEEDS_LOGITS,
    LABEL_FN_USES_TOPK,
    LABEL_FUNCTIONS,
)
from metaue_src.model import MetaUEMLP
from metaue_src.train import _set_seed, train_metaue
from logit_magnitude_src.logit_magnitude import (
    is_correct,
    is_correct_qa,
    find_optimal_threshold_youden,
    _load_parquet_sharded,
)
from logit_magnitude_src.metrics import auroc as _auroc, aurac as _aurac


SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val/optimal_auroc", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [3e-3, 1e-3, 3e-4]},
        "dropout": {"values": [0.3]},
        "batch_size": {"values": [64, 128, 256]},
    },
}

# Module-level caches
_LOADED_DATA: Dict[str, Any] = {}
_RUN_COUNTER = {"n": 0}
_LABEL_CACHE: Dict[tuple, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Inlined helpers
# ---------------------------------------------------------------------------

def _detect_task_type(dataset: str) -> str:
    """Use ROUGE-L correctness for open QA datasets."""
    qa_datasets = {"coqa", "newsqa", "emrqa", "nq", "triviaqa"}
    return "qa" if dataset.lower() in qa_datasets else "mcq"


def _build_paths(
    inference_dir: str,
    dataset: str,
    model_tag: str,
    max_input_len: str,
    pe_method: str,
    n_responses: int,
    data_split: str,
) -> tuple:
    """Return (result_stem, logits_stem) for _load_parquet_sharded."""
    base = Path(inference_dir) / dataset
    prefix = f"{dataset}_{max_input_len}_{model_tag}_{pe_method}_nres-{n_responses}"
    result_stem = base / f"{prefix}_set-{data_split}"
    logits_stem = base / f"{prefix}_logits_set-{data_split}"
    return result_stem, logits_stem


# ---------------------------------------------------------------------------
# Label caching
# ---------------------------------------------------------------------------

def _scan_logits_column(stem: Path, response_idx: int) -> pl.DataFrame:
    col = f"gen_topk_logits_{response_idx}"
    if stem.exists():
        return pl.scan_parquet(stem).select([col]).collect(engine="streaming")
    shards = sorted(stem.parent.glob(f"{stem.name}-*-of-*.parquet"))
    if shards:
        return pl.scan_parquet(shards).select([col]).collect(engine="streaming")
    raise FileNotFoundError(f"Logits parquet not found at {stem}")


def _label_disk_path(label_save_dir: str, label_fn_name: str, topk_worst: int,
                     split: str) -> Path:
    base = Path(label_save_dir)
    uses_topk = LABEL_FN_USES_TOPK.get(label_fn_name, False)
    if uses_topk:
        filename = f"{label_fn_name}_topk{topk_worst}_{split}_reg.npz"
    else:
        filename = f"{label_fn_name}_{split}_reg.npz"
    return base / filename


def _get_or_compute_labels(
    split: str,
    label_fn_name: str,
    topk_worst: int,
    response_idx: int,
    result_df: pl.DataFrame,
    correct_fn: Callable,
    label_save_dir: Optional[str] = None,
    **extra_label_fn_kwargs,
) -> Dict[str, Any]:
    """Return label arrays for this (label_fn, split) key.

    Priority: in-memory cache → disk cache → compute (then save to disk).
    Labels are always raw continuous float32 scores for MSE regression.
    """
    key = (label_fn_name, topk_worst, response_idx, split)

    if key in _LABEL_CACHE:
        print(f"[LabelCache] hit (memory) {key}")
        return _LABEL_CACHE[key]

    disk_path = None
    if label_save_dir is not None:
        disk_path = _label_disk_path(label_save_dir, label_fn_name, topk_worst, split)
        if disk_path.exists():
            print(f"[LabelCache] hit (disk) {key} ← {disk_path}")
            data = np.load(disk_path, allow_pickle=False)
            version = np.asarray(data.get("label_method_version", None)).item() \
                if "label_method_version" in data else None
            if isinstance(version, str) and version.startswith("regression_v1::"):
                entry: Dict[str, Any] = {
                    "labels": data["labels"].astype(np.float32),
                    "valid_indices": data["valid_indices"].astype(np.int32),
                }
                _LABEL_CACHE[key] = entry
                return entry
            print(f"[LabelCache] stale cache — recomputing {key}")

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[LabelCache] miss {key} — computing labels ...")

    needs_logits = LABEL_FN_NEEDS_LOGITS.get(label_fn_name, False)
    uses_topk = LABEL_FN_USES_TOPK.get(label_fn_name, False)

    logits_df = None
    if needs_logits:
        logits_df = _scan_logits_column(
            _LOADED_DATA["logits_stems"][split], response_idx
        )

    label_kwargs: Dict[str, Any] = {}
    if uses_topk:
        label_kwargs["topk_worst"] = topk_worst
    label_kwargs.update(extra_label_fn_kwargs)

    label_fn = LABEL_FUNCTIONS[label_fn_name]
    result_rows = result_df.to_dicts()
    logits_iter: Any = (
        logits_df.iter_rows(named=True) if logits_df is not None
        else iter([None] * len(result_rows))
    )

    labels_list: list = []
    valid_indices_list: list = []
    for i, (row, logits_row) in enumerate(zip(result_rows, logits_iter)):
        lbl = label_fn(row, logits_row, response_idx=response_idx,
                       threshold=None, **label_kwargs)
        if lbl is None:
            continue
        valid_indices_list.append(i)
        labels_list.append(float(lbl))

    if logits_df is not None:
        del logits_df
        gc.collect()

    entry = {
        "labels": np.array(labels_list, dtype=np.float32),
        "valid_indices": np.array(valid_indices_list, dtype=np.int32),
    }

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[LabelCache] done {key} — {len(entry['labels'])} labels, "
          f"peak RSS delta: {(rss_after - rss_before):+d} KB")

    if disk_path is not None:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            disk_path,
            labels=entry["labels"],
            valid_indices=entry["valid_indices"],
            label_method_version=np.array(f"regression_v1::{label_fn_name}", dtype=object),
        )
        print(f"[LabelCache] saved → {disk_path}")

    _LABEL_CACHE[key] = entry
    return entry


def _compute_gt_correct_flags(
    result_df: pl.DataFrame,
    valid_indices: np.ndarray,
    correct_fn: Callable,
) -> List[bool]:
    """GT correctness for the valid_indices subset."""
    labels = (result_df.get_column("label").to_list()
              if "label" in result_df.columns else [None] * result_df.height)
    preds = (result_df.get_column("answer_0").to_list()
             if "answer_0" in result_df.columns else [None] * result_df.height)
    all_flags = [correct_fn(lbl, pred) for lbl, pred in zip(labels, preds)]
    return [all_flags[i] for i in valid_indices]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_data_loaded(fixed_args: argparse.Namespace):
    """Load parquets and pre-encoded embeddings once; subsequent calls are no-ops."""
    if "train_embs" in _LOADED_DATA:
        return

    dataset = fixed_args.dataset
    model_path = fixed_args.model_path
    model_tag = _model_tag_from_path(model_path)
    emb_model_name = fixed_args.embedding_model_name
    inference_dir = fixed_args.inference_dir
    max_input_len = DEFAULT_CONFIG["max_input_len"]
    pe_method = DEFAULT_CONFIG["pe_method"]
    n_responses = DEFAULT_CONFIG["n_responses"]
    train_split = DEFAULT_CONFIG["train_split"]
    val_split = DEFAULT_CONFIG["val_split"]
    test_split = DEFAULT_CONFIG["test_split"]
    emb_parquet_dir = DEFAULT_CONFIG["embedding_parquet_dir"]

    print("Loading inference parquets ...")
    logits_stems: Dict[str, Any] = {}
    for split in (train_split, val_split, test_split):
        result_stem, logits_stem = _build_paths(
            inference_dir, dataset, model_tag, max_input_len, pe_method, n_responses, split
        )
        _LOADED_DATA[f"{split}_result"] = _load_parquet_sharded(result_stem)
        logits_stems[split] = logits_stem
    _LOADED_DATA["logits_stems"] = logits_stems

    print(f"  Train: {_LOADED_DATA[f'{train_split}_result'].height} | "
          f"Val: {_LOADED_DATA[f'{val_split}_result'].height} | "
          f"Test: {_LOADED_DATA[f'{test_split}_result'].height}")

    print("Loading pre-encoded embedding parquets ...")
    for split in (train_split, val_split, test_split):
        path = embedding_parquet_path(emb_parquet_dir, dataset, split, emb_model_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding parquet not found: {path}\n"
                f"Run first:  python encode_metaue.py "
                f"--dataset {dataset} "
                f"--embedding_model_name {emb_model_name}"
            )
        print(f"  [{split}] {path}")
        _LOADED_DATA[f"{split}_embs"] = load_embedding_parquet(
            emb_parquet_dir, dataset, split, emb_model_name
        )

    task_type = _detect_task_type(dataset)
    _LOADED_DATA["correct_fn"] = is_correct_qa if task_type == "qa" else is_correct
    _LOADED_DATA["model_tag"] = model_tag
    _LOADED_DATA["dataset"] = dataset
    _LOADED_DATA["train_split"] = train_split
    _LOADED_DATA["val_split"] = val_split
    _LOADED_DATA["test_split"] = test_split

    print("Data loaded and cached.")


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def _build_datasets(fixed_args: argparse.Namespace, label_fn_kwargs: Dict[str, Any]):
    """Build train/val/test MetaUEDatasets using label cache (disk-backed)."""
    _ensure_data_loaded(fixed_args)

    label_fn_name = "logit_magnitude"
    model_tag = _LOADED_DATA["model_tag"]
    dataset = _LOADED_DATA["dataset"]
    train_split = _LOADED_DATA["train_split"]
    val_split = _LOADED_DATA["val_split"]
    test_split = _LOADED_DATA["test_split"]
    correct_fn = _LOADED_DATA["correct_fn"]

    topk_worst = fixed_args.topk_worst
    response_idx = DEFAULT_CONFIG["response_idx"]
    embedding_tag = fixed_args.embedding_model_name.split("/")[-1]
    label_save_dir = str(Path(fixed_args.log_dir) / "labels" / f"{model_tag}_{dataset}_{embedding_tag}")

    train_entry = _get_or_compute_labels(
        train_split, label_fn_name, topk_worst, response_idx,
        _LOADED_DATA[f"{train_split}_result"], correct_fn,
        label_save_dir=label_save_dir,
        **label_fn_kwargs,
    )
    val_entry = _get_or_compute_labels(
        val_split, label_fn_name, topk_worst, response_idx,
        _LOADED_DATA[f"{val_split}_result"], correct_fn,
        label_save_dir=label_save_dir,
        **label_fn_kwargs,
    )
    test_entry = _get_or_compute_labels(
        test_split, label_fn_name, topk_worst, response_idx,
        _LOADED_DATA[f"{test_split}_result"], correct_fn,
        label_save_dir=label_save_dir,
        **label_fn_kwargs,
    )

    train_embs = _LOADED_DATA[f"{train_split}_embs"]
    val_embs = _LOADED_DATA[f"{val_split}_embs"]
    test_embs = _LOADED_DATA[f"{test_split}_embs"]

    # Min-max normalisation using training-set statistics
    train_labels_raw = train_entry["labels"].astype(np.float32)
    label_min = float(train_labels_raw.min())
    label_max = float(train_labels_raw.max())
    denom = max(label_max - label_min, 1e-8)
    print(f"[Norm] label min={label_min:.4f}  max={label_max:.4f}  denom={denom:.4f}")

    def _normalise(arr: np.ndarray) -> np.ndarray:
        return np.clip((arr.astype(np.float32) - label_min) / denom, 0.0, 1.0)

    train_dataset = MetaUEDataset.from_labels(
        _normalise(train_entry["labels"]), train_entry["valid_indices"], train_embs,
        label_norm_params=(label_min, label_max),
    )
    val_dataset = MetaUEDataset.from_labels(
        _normalise(val_entry["labels"]), val_entry["valid_indices"], val_embs,
        label_norm_params=(label_min, label_max),
    )
    test_dataset = MetaUEDataset.from_labels(
        _normalise(test_entry["labels"]), test_entry["valid_indices"], test_embs,
        label_norm_params=(label_min, label_max),
    )

    return train_dataset, val_dataset, test_dataset, val_entry, test_entry


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_uq_metrics(
    uncertainty: List[float],
    correct_flags: List[bool],
    n_bootstrap: int = 1000,
    seed: int = 0,
    opt_thresh: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap AUROC, AUPRC, AURAC, and threshold-sensitive UQ metrics."""
    from sklearn.metrics import balanced_accuracy_score, roc_curve as _roc_curve

    rng = np.random.default_rng(seed)
    n = len(uncertainty)
    unc_arr = np.array(uncertainty, dtype=np.float64)
    cor_arr = np.array(correct_flags, dtype=bool)

    results: Dict[str, List[float]] = {
        "auroc": [], "aurac": [], "balanced_acc": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        u_b = unc_arr[idx]
        c_b = cor_arr[idx]
        if c_b.all() or not c_b.any():
            continue

        y_true = np.where(c_b, 0, 1)
        try:
            results["auroc"].append(float(_auroc(y_true, u_b)))
            results["aurac"].append(float(_aurac(c_b.astype(np.float64), u_b)))
        except Exception:
            pass

        try:
            if opt_thresh is not None:
                thresh = opt_thresh
            else:
                fpr_b, tpr_b, threshs_b = _roc_curve(y_true, u_b)
                thresh = float(threshs_b[int(np.argmax(tpr_b - fpr_b))]) if len(threshs_b) > 0 else 0.5
            y_pred_b = (u_b >= thresh).astype(int)
            results["balanced_acc"].append(float(balanced_accuracy_score(y_true, y_pred_b)))
        except Exception:
            pass

    summary: Dict[str, Dict[str, float]] = {}
    for metric, vals in results.items():
        finite_vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not finite_vals:
            summary[metric] = {"mean": float("nan"), "std": float("nan"),
                               "ci_low": float("nan"), "ci_high": float("nan")}
        else:
            arr = np.array(finite_vals)
            summary[metric] = {
                "mean": round(float(arr.mean()), 3),
                "std": round(float(arr.std()), 3),
                "ci_low": round(float(np.percentile(arr, 2.5)), 3),
                "ci_high": round(float(np.percentile(arr, 97.5)), 3),
            }
    return summary


# ---------------------------------------------------------------------------
# Post-sweep pipeline
# ---------------------------------------------------------------------------

def _find_best_sweep(sweeps_dir: Path) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_auroc = -1.0
    for metrics_path in sorted(sweeps_dir.glob("*/metrics.json")):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            val_auroc = data.get("best_val", {}).get("auroc", -1.0)
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best = dict(data)
                best["_metrics_path"] = str(metrics_path)
                best["_sweep_subdir"] = metrics_path.parent.name
        except Exception as e:
            print(f"[PostSweep] skipping {metrics_path}: {e}")
    return best


def _run_post_sweep_pipeline(
    args: argparse.Namespace,
    sweep_group: str,
    sweep_id: str,
) -> None:
    """After WandB sweep: find best hyperparams, train 5 seeds, save report."""
    label_fn_name = "logit_magnitude"
    study_dir = Path(args.log_dir) / sweep_group
    sweeps_dir = study_dir / "sweeps"

    print(f"\n{'='*60}")
    print(f"[PostSweep] Finding best sweep in {sweeps_dir} ...")
    best = _find_best_sweep(sweeps_dir)
    if best is None:
        print("[PostSweep] No completed sweep runs found — skipping post-sweep training.")
        return

    optimal_payload = {
        "study": sweep_group,
        "sweep_id": sweep_id,
        "best_sweep_subdir": f"sweeps/{best['_sweep_subdir']}",
        "hyperparams": best.get("hyperparams", {}),
        "best_val": best.get("best_val", {}),
        "test": best.get("test", {}),
    }
    optimal_path = study_dir / "optimal_sweep.json"
    with open(optimal_path, "w") as f:
        json.dump(optimal_payload, f, indent=2)
    print(f"[PostSweep] Optimal hyperparams saved → {optimal_path}")

    optimal_hp = best.get("hyperparams", {})
    per_seed_results: List[Dict[str, Any]] = []

    for seed in args.train_seeds:
        print(f"\n[PostSweep] Training seed={seed} ...")
        seed_dir = study_dir / "trained" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        _set_seed(seed)
        train_dataset, val_dataset, test_dataset, val_entry, test_entry = \
            _build_datasets(args, {})

        correct_fn = _LOADED_DATA["correct_fn"]
        val_gt_flags = _compute_gt_correct_flags(
            _LOADED_DATA[f"{_LOADED_DATA['val_split']}_result"],
            val_entry["valid_indices"], correct_fn,
        )
        test_gt_flags = _compute_gt_correct_flags(
            _LOADED_DATA[f"{_LOADED_DATA['test_split']}_result"],
            test_entry["valid_indices"], correct_fn,
        )

        config = dict(DEFAULT_CONFIG)
        config.update(optimal_hp)
        config["seed"] = seed
        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        config["dataset"] = args.dataset
        config["model_path"] = args.model_path
        config["model_tag"] = _model_tag_from_path(args.model_path)
        config["embedding_model_name"] = args.embedding_model_name
        config["regression_loss"] = args.regression_loss
        config["rank_lambda"] = args.rank_lambda
        config["run_name"] = f"{sweep_group}_seed-{seed}"
        config["wandb_enabled"] = False
        config["checkpoint_dir"] = str(seed_dir)

        embed_dim = train_dataset.embeddings.shape[1]
        model = MetaUEMLP(embed_dim, dropout=config["dropout"])

        trained_model, best_val_metrics = train_metaue(
            model, train_dataset, val_dataset, config, wandb_run=None,
            val_gt_correct_flags=val_gt_flags,
        )

        val_metrics_for_cal = evaluate_metaue(
            trained_model, val_dataset, device=config["device"], batch_size=256
        )
        val_uncertainty = val_metrics_for_cal["per_sample"]
        opt_thresh = find_optimal_threshold_youden(val_uncertainty, val_gt_flags)

        test_metrics = evaluate_metaue(
            trained_model, test_dataset, device=config["device"], batch_size=256,
            gt_correct_flags=test_gt_flags,
        )

        test_uncertainty = test_metrics["per_sample"]
        bootstrapped = _bootstrap_uq_metrics(
            test_uncertainty, test_gt_flags, seed=seed, opt_thresh=opt_thresh
        )

        seed_result = {
            "seed": seed,
            "best_val": best_val_metrics,
            "test_point": {k: v for k, v in test_metrics.items() if k != "per_sample"},
            "test_bootstrapped": bootstrapped,
        }
        per_seed_results.append(seed_result)

        torch.save(trained_model.state_dict(), seed_dir / "weights.pt")
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(seed_result, f, indent=2)
        print(f"[PostSweep] seed={seed} test auroc={test_metrics.get('auroc', float('nan')):.4f}")

    # Aggregate report
    metric_keys = ["auroc", "aurac", "balanced_acc"]
    averaged: Dict[str, Dict[str, float]] = {}
    for mk in metric_keys:
        means = [r["test_bootstrapped"][mk]["mean"] for r in per_seed_results
                 if mk in r["test_bootstrapped"] and np.isfinite(r["test_bootstrapped"][mk]["mean"])]
        if means:
            averaged[mk] = {
                "mean_of_bootstrapped_means": float(np.mean(means)),
                "std_across_seeds": float(np.std(means)),
            }
        else:
            averaged[mk] = {"mean_of_bootstrapped_means": float("nan"), "std_across_seeds": float("nan")}

    report = {
        "study": sweep_group,
        "optimal_hyperparams": optimal_payload["hyperparams"],
        "seeds": args.train_seeds,
        "per_seed": per_seed_results,
        "averaged": averaged,
    }
    report_path = study_dir / "trained" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[PostSweep] Report saved → {report_path}")
    for mk in metric_keys:
        avg = averaged[mk]
        print(f"  {mk}: {avg['mean_of_bootstrapped_means']:.4f} ± {avg['std_across_seeds']:.4f}")

    # Save the best seed's weights as best_model.pt and write a path marker
    if per_seed_results:
        best_result = max(per_seed_results, key=lambda r: r["test_point"].get("auroc", -1.0))
        best_seed = best_result["seed"]
        best_src = study_dir / "trained" / f"seed_{best_seed}" / "weights.pt"
        best_model_path = study_dir / "trained" / "best_model.pt"
        best_state = torch.load(best_src, map_location="cpu", weights_only=True)
        torch.save(best_state, best_model_path)
        marker = Path(args.log_dir) / "last_best_model_path.txt"
        marker.write_text(str(best_model_path.resolve()))
        print(f"\n[PostSweep] Best model (seed={best_seed}, "
              f"auroc={best_result['test_point'].get('auroc', float('nan')):.4f})"
              f" → {best_model_path}")
        print(f"[PostSweep] Path written to {marker}")


# ---------------------------------------------------------------------------
# Sweep train function
# ---------------------------------------------------------------------------

def _make_sweep_train_fn(fixed_args: argparse.Namespace, sweep_metadata: Dict[str, Any]):
    """Return a sweep_train function closed over fixed_args and sweep_metadata."""

    def sweep_train():
        import traceback as _tb

        _RUN_COUNTER["n"] += 1
        run_idx = _RUN_COUNTER["n"]

        label_fn_name = "logit_magnitude"
        model_tag_pre = _model_tag_from_path(fixed_args.model_path)
        dataset_pre = fixed_args.dataset
        embedding_tag_pre = fixed_args.embedding_model_name.split("/")[-1]
        sweep_group = f"{model_tag_pre}_{dataset_pre}_{label_fn_name}_{embedding_tag_pre}"
        run_name = f"{sweep_group}_sweep-{run_idx}"

        run = wandb.init(name=run_name)
        try:
            sweep_cfg = dict(wandb.config)

            config = dict(DEFAULT_CONFIG)
            config.update(sweep_cfg)
            config["seed"] = random.randint(0, 10000)
            config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
            config["dataset"] = fixed_args.dataset
            config["model_path"] = fixed_args.model_path
            config["model_tag"] = model_tag_pre
            config["embedding_model_name"] = fixed_args.embedding_model_name
            config["regression_loss"] = fixed_args.regression_loss
            config["rank_lambda"] = fixed_args.rank_lambda
            config["run_name"] = run_name
            config["wandb_enabled"] = False
            run_dir = Path(fixed_args.log_dir) / sweep_group / "sweeps" / f"sweep-{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config["checkpoint_dir"] = str(run_dir)

            _set_seed(config["seed"])

            train_dataset, val_dataset, test_dataset, val_entry, test_entry = \
                _build_datasets(fixed_args, {})

            correct_fn = _LOADED_DATA["correct_fn"]
            val_gt_flags = _compute_gt_correct_flags(
                _LOADED_DATA[f"{_LOADED_DATA['val_split']}_result"],
                val_entry["valid_indices"], correct_fn,
            )
            test_gt_flags = _compute_gt_correct_flags(
                _LOADED_DATA[f"{_LOADED_DATA['test_split']}_result"],
                test_entry["valid_indices"], correct_fn,
            )

            embed_dim = train_dataset.embeddings.shape[1]
            model = MetaUEMLP(embed_dim, dropout=config["dropout"])

            trained_model, best_val_metrics = train_metaue(
                model, train_dataset, val_dataset, config, wandb_run=run,
                val_gt_correct_flags=val_gt_flags,
            )

            val_metrics_for_cal = evaluate_metaue(
                trained_model, val_dataset, device=config["device"], batch_size=256
            )

            test_metrics = evaluate_metaue(
                trained_model, test_dataset, device=config["device"], batch_size=256,
                gt_correct_flags=test_gt_flags,
            )

            run.log({
                "val/optimal_auroc": best_val_metrics.get("auroc", float("nan")),
                "val/optimal_aurac": best_val_metrics.get("aurac", float("nan")),
                "val/optimal_balanced_acc": best_val_metrics.get("balanced_acc", float("nan")),
                "test/auroc": test_metrics.get("auroc", float("nan")),
                "test/aurac": test_metrics.get("aurac", float("nan")),
                "test/balanced_acc": test_metrics.get("balanced_acc", float("nan")),
            })

            weights_path = run_dir / "weights.pt"
            torch.save(trained_model.state_dict(), weights_path)

            hyperparams = {k: sweep_cfg.get(k) for k in ["lr", "dropout", "batch_size"]}
            hyperparams["label_fn_name"] = label_fn_name
            hyperparams["topk_worst"] = fixed_args.topk_worst
            metrics_payload = {
                "sweep_metadata": {
                    **sweep_metadata,
                    "run_idx": run_idx,
                    "run_id": run.id,
                    "run_name": f"sweep-{run_idx}",
                    "seed": config["seed"],
                },
                "hyperparams": hyperparams,
                "best_val": best_val_metrics,
                "test": {k: v for k, v in test_metrics.items() if k != "per_sample"},
            }
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics_payload, f, indent=2)

            print(
                f"[sweep-{run_idx}] Saved → {run_dir}  |  "
                f"val_auroc={best_val_metrics.get('auroc', float('nan')):.4f}  "
                f"test_auroc={test_metrics.get('auroc', float('nan')):.4f}"
            )

        except Exception:
            print(f"\n[sweep-{run_idx}] ERROR — full traceback below:")
            _tb.print_exc()
            raise
        finally:
            run.finish()

    return sweep_train


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WandB grid search for MetaUE (MLP-only)")
    parser.add_argument("--dataset", type=str, default="coqa")
    parser.add_argument("--model_path", type=str, default="google/gemma-4-e4b-it",
                        help="Model path used to name inference parquets")
    parser.add_argument("--embedding_model_name", type=str,
                        default=DEFAULT_CONFIG["embedding_model_name"])
    parser.add_argument("--inference_dir", type=str, default="data/llm-inference",
                        help="Root directory containing inference parquets")
    parser.add_argument("--topk_worst", type=int, default=DEFAULT_CONFIG["topk_worst"],
                        help="Top-k worst parameter for logit_magnitude label fn")
    parser.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Random seeds for post-sweep 5-seed training")
    parser.add_argument("--project", type=str, default="metaue-demo")
    parser.add_argument("--log_dir", type=str, default="data/metaue_sweep",
                        help="Root directory for labels, weights, and metrics")
    parser.add_argument("--gpu_ids", type=lambda s: [int(x) for x in s.split(",")],
                        default=[0])
    parser.add_argument("--regression_loss", type=str, default="mse",
                        choices=["mse", "rank", "hybrid"])
    parser.add_argument("--rank_lambda", type=float, default=0.5)
    args = parser.parse_args()

    label_fn_name = "logit_magnitude"
    model_tag = _model_tag_from_path(args.model_path)
    embedding_tag = args.embedding_model_name.split("/")[-1]
    sweep_group = f"{model_tag}_{args.dataset}_{label_fn_name}_{embedding_tag}"
    sweep_name = f"{sweep_group}_metaue"

    sweep_cfg = dict(SWEEP_CONFIG)
    sweep_cfg["parameters"] = dict(SWEEP_CONFIG["parameters"])
    sweep_cfg["parameters"]["label_fn_name"] = {"value": label_fn_name}
    sweep_cfg["parameters"]["dataset"] = {"value": args.dataset}
    sweep_cfg["parameters"]["model_path"] = {"value": args.model_path}
    sweep_cfg["parameters"]["embedding_model_name"] = {"value": args.embedding_model_name}
    sweep_cfg["name"] = sweep_name

    sweep_id = wandb.sweep(sweep_cfg, project=args.project)

    sweep_metadata = {
        "sweep_id": sweep_id,
        "project": args.project,
        "dataset": args.dataset,
        "model_tag": model_tag,
        "label_fn_name": label_fn_name,
        "embedding_model_name": args.embedding_model_name,
        "topk_worst": args.topk_worst,
        "log_dir": args.log_dir,
    }

    sweep_train_fn = _make_sweep_train_fn(args, sweep_metadata)
    wandb.agent(sweep_id, function=sweep_train_fn)

    _run_post_sweep_pipeline(args, sweep_group, sweep_id)


if __name__ == "__main__":
    main()
