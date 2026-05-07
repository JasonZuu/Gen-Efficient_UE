import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from metaue_src.dataset import MetaUEDataset, collate_fn_embeddings, collate_fn_tokenized
from metaue_src.evaluate import evaluate_metaue
from metaue_src.model import MetaUEModel, MetaUEMLP


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _encode_all_prompts(
    model: MetaUEModel,
    prompts: list,
    device: str,
    batch_size: int = 16,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode all prompts with the frozen embedding model. Returns (N, D) on CPU."""
    model.eval()
    model.embedding_model.to(device)
    all_embeddings = []
    with tqdm(total=len(prompts), desc="Encoding prompts", unit="prompt", leave=False) as pbar:
        for i in range(0, len(prompts), batch_size):
            batch_texts = prompts[i : i + batch_size]
            encoded = model.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            emb = model.encode(input_ids, attention_mask).cpu()
            all_embeddings.append(emb)
            pbar.update(len(batch_texts))
    return torch.cat(all_embeddings, dim=0)


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def train_metaue(
    model,
    train_dataset: MetaUEDataset,
    val_dataset: Optional[MetaUEDataset],
    config: Dict[str, Any],
    wandb_run: Optional[Any] = None,
    val_gt_correct_flags: Optional[list] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Step-based training loop for MetaUE.

    LR schedule: linear warmup for first 5% of steps, cosine annealing for the rest.
    Validation every val_steps steps. Early stopping after 10% of total steps without
    improvement on val AUROC.

    val_gt_correct_flags: if provided, validation AUROC uses real GT correctness.

    Returns (best_model, best_val_metrics).
    """
    device = config["device"]
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    weight_decay = config["weight_decay"]
    val_steps = config.get("val_steps", 200)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    collate = collate_fn_embeddings if train_dataset.use_embeddings else collate_fn_tokenized
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(0.05 * total_steps))
    patience_steps = int(0.1 * total_steps)

    loss_kind = config.get("regression_loss", "mse")
    rank_lam = float(config.get("rank_lambda", 0.5))
    _mse_fn = nn.MSELoss()

    def _rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]
        if B < 2:
            return pred.new_zeros(())
        pi = pred.unsqueeze(1) - pred.unsqueeze(0)
        ti = target.unsqueeze(1) - target.unsqueeze(0)
        mask = ti.abs() > 1e-6
        if mask.sum() == 0:
            return pred.new_zeros(())
        loss = F.softplus(-torch.sign(ti) * pi)
        return loss[mask].mean()

    if loss_kind == "mse":
        criterion = _mse_fn
        print(f"[Train] Using MSE loss")
    elif loss_kind == "rank":
        criterion = _rank_loss
        print(f"[Train] Using RankNet pairwise loss")
    else:  # hybrid
        def criterion(pred, target):  # type: ignore[misc]
            return _mse_fn(pred, target) + rank_lam * _rank_loss(pred, target)
        print(f"[Train] Using hybrid MSE + {rank_lam}×rank loss")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=_make_lr_lambda(warmup_steps, total_steps))

    model.to(device)
    checkpoint_path = checkpoint_dir / f"{config.get('run_name', 'metaue')}_best.pt"

    best_val_auroc = -1.0
    best_step = 0
    best_val_metrics: Dict[str, Any] = {}
    steps_no_improve = 0
    global_step = 0
    last_loss = float("nan")

    pbar = tqdm(total=total_steps, desc="MetaUE", dynamic_ncols=True, unit="step")

    def _run_validation() -> bool:
        nonlocal best_val_auroc, best_step, best_val_metrics, steps_no_improve

        model.eval()
        val_metrics = evaluate_metaue(
            model, val_dataset, device=device, batch_size=batch_size * 2,
            gt_correct_flags=val_gt_correct_flags,
        )
        model.train()

        val_auroc = val_metrics["auroc"]
        val_aurac = val_metrics["aurac"]
        val_balanced_acc = val_metrics.get("balanced_acc", float("nan"))

        pbar.write(
            f"[step {global_step:6d}] val/auroc={val_auroc:.4f}  "
            f"val/aurac={val_aurac:.4f}  val/balanced_acc={val_balanced_acc:.4f}"
            + (f"  [best={best_val_auroc:.4f} @{best_step}]" if best_step > 0 else "")
        )

        if wandb_run is not None:
            wandb_run.log(
                {"val/auroc": val_auroc, "val/aurac": val_aurac,
                 "val/balanced_acc": val_balanced_acc},
                step=global_step,
            )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_step = global_step
            best_val_metrics = {k: v for k, v in val_metrics.items() if k != "per_sample"}
            steps_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            steps_no_improve += val_steps

        pbar.set_postfix(
            {"loss": f"{last_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}",
             "v_auc": f"{val_auroc:.4f}"},
            refresh=True,
        )

        return steps_no_improve >= patience_steps and global_step > warmup_steps

    stop = False
    try:
        for _ in range(epochs):
            if stop:
                break
            model.train()
            for batch in train_loader:
                if stop or global_step >= total_steps:
                    stop = True
                    break

                optimizer.zero_grad()
                if train_dataset.use_embeddings:
                    emb, labels = batch
                    emb = emb.to(device)
                    logits = model.forward_from_embedding(emb)
                else:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    emb = model.encode(input_ids, attention_mask)
                    logits = model.mlp(emb)

                labels = labels.to(device)
                loss = criterion(logits.squeeze(-1), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                global_step += 1
                last_loss = loss.item()
                current_lr = scheduler.get_last_lr()[0]

                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{last_loss:.4f}", "lr": f"{current_lr:.2e}"},
                    refresh=False,
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {"train/loss": last_loss, "train/lr": current_lr},
                        step=global_step,
                    )

                if val_dataset is not None and len(val_dataset) > 0 and global_step % val_steps == 0:
                    if _run_validation():
                        pbar.write(
                            f"Early stopping at step {global_step} "
                            f"(no improvement for {steps_no_improve} steps, "
                            f"patience={patience_steps})"
                        )
                        stop = True
                        break
    finally:
        pbar.close()

    # Final validation if the last step wasn't a val checkpoint
    if val_dataset is not None and len(val_dataset) > 0 and global_step % val_steps != 0:
        model.eval()
        val_metrics = evaluate_metaue(
            model, val_dataset, device=device, batch_size=batch_size * 2,
            gt_correct_flags=val_gt_correct_flags,
        )
        model.train()
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_step = global_step
            best_val_metrics = {k: v for k, v in val_metrics.items() if k != "per_sample"}
            torch.save(model.state_dict(), checkpoint_path)

    # Restore best weights
    if val_dataset is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Restored best model from step {best_step} (val_auroc={best_val_auroc:.4f})")

    return model, best_val_metrics
