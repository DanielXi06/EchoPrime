from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


def _set_cuda_visible_devices_from_argv() -> None:
    for index, arg in enumerate(sys.argv):
        if arg == "--gpu-id" and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return
        if arg.startswith("--gpu-id="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_set_cuda_visible_devices_from_argv()

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable

from echo_prime.video_classifier import EchoPrimeBinaryClassifier
from echo_prime.video_data import EchoPrimeVideoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a video-level binary classifier from EchoPrime encoder weights."
    )
    parser.add_argument("--train-csv", required=True, help="CSV containing training videos.")
    parser.add_argument("--val-csv", default=None, help="CSV containing validation videos.")
    parser.add_argument("--data-root", default=None, help="Root for relative video paths.")
    parser.add_argument("--path-column", default="video_path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument(
        "--weights-path",
        default="model_data/weights/echo_prime_encoder.pt",
        help="Path to echo_prime_encoder.pt.",
    )
    parser.add_argument("--output-dir", default="runs/echo_prime_binary")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-clips", type=int, default=5)
    parser.add_argument(
        "--eval-aggregation",
        choices=["mean", "max", "topk_mean"],
        default="mean",
        help="How to merge clip logits into one video logit.",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--hidden-dims",
        default="256",
        help="Comma-separated MLP hidden dimensions. Use empty string for linear head.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=4e-5,
        help="EchoPrime encoder LR when not frozen; mirrors the paper pretraining LR.",
    )
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="AdamW weight decay; mirrors the paper setting.",
    )
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument(
        "--best-metric",
        choices=["val_loss", "val_acc", "val_auc"],
        default="val_auc",
    )
    parser.add_argument(
        "--pos-weight",
        default="auto",
        help="'auto', 'none', or a float passed to BCEWithLogitsLoss.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Physical GPU id to expose before torch imports, e.g. --gpu-id 1.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-plot-curves", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    val_csv = args.val_csv or args.train_csv
    train_dataset = EchoPrimeVideoDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        path_column=args.path_column,
        label_column=args.label_column,
        split_column=args.split_column,
        split_value=args.train_split,
        mode="train",
        eval_clips=args.eval_clips,
    )
    val_dataset = EchoPrimeVideoDataset(
        csv_path=val_csv,
        data_root=args.data_root,
        path_column=args.path_column,
        label_column=args.label_column,
        split_column=args.split_column,
        split_value=args.val_split,
        mode="eval",
        eval_clips=args.eval_clips,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def positive_weight(labels: list[float], arg_value: str, device: torch.device) -> torch.Tensor | None:
    value = str(arg_value).strip().lower()
    if value in {"none", "false", "0"}:
        return None
    if value == "auto":
        positives = sum(1 for label in labels if label == 1.0)
        negatives = sum(1 for label in labels if label == 0.0)
        if positives == 0 or negatives == 0:
            return None
        return torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


def build_optimizer(
    model: EchoPrimeBinaryClassifier,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.freeze_encoder:
        params = [{"params": model.classifier.parameters(), "lr": args.head_lr}]
    else:
        params = [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ]
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def aggregate_clip_logits(
    logits: torch.Tensor,
    method: str,
    topk: int,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits of shape B x K, got {tuple(logits.shape)}")
    if method == "mean":
        return logits.mean(dim=1)
    if method == "max":
        return logits.max(dim=1).values
    if method == "topk_mean":
        k = min(max(1, topk), logits.shape[1])
        return torch.topk(logits, k=k, dim=1).values.mean(dim=1)
    raise ValueError(f"Unsupported aggregation method: {method}")


def forward_batch(
    model: EchoPrimeBinaryClassifier,
    video: torch.Tensor,
    aggregation: str,
    topk: int,
) -> torch.Tensor:
    if video.ndim == 5:
        return model(video)
    if video.ndim == 6:
        batch_size, num_clips, channels, frames, height, width = video.shape
        flat_video = video.reshape(batch_size * num_clips, channels, frames, height, width)
        flat_logits = model(flat_video)
        clip_logits = flat_logits.reshape(batch_size, num_clips)
        return aggregate_clip_logits(clip_logits, aggregation, topk)
    raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")


def binary_auc(labels: list[float], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label == 1.0)
    negatives = sum(1 for label in labels if label == 0.0)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores))
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[np.asarray(labels) == 1.0].sum()
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def compute_metrics(
    labels: list[float],
    logits: list[float],
    loss: float,
    threshold: float,
) -> dict[str, float]:
    probs = [1.0 / (1.0 + math.exp(-logit)) for logit in logits]
    preds = [1.0 if prob >= threshold else 0.0 for prob in probs]
    total = len(labels)
    correct = sum(1 for y, pred in zip(labels, preds) if y == pred)
    tp = sum(1 for y, pred in zip(labels, preds) if y == 1.0 and pred == 1.0)
    tn = sum(1 for y, pred in zip(labels, preds) if y == 0.0 and pred == 0.0)
    fp = sum(1 for y, pred in zip(labels, preds) if y == 0.0 and pred == 1.0)
    fn = sum(1 for y, pred in zip(labels, preds) if y == 1.0 and pred == 0.0)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else float("nan")
    )
    return {
        "loss": loss,
        "acc": correct / total if total else float("nan"),
        "auc": binary_auc(labels, probs),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def run_epoch(
    model: EchoPrimeBinaryClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    aggregation: str,
    topk: int,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    desc: str | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    all_labels: list[float] = []
    all_logits: list[float] = []
    scaler_enabled = amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    progress = tqdm(loader, total=len(loader), desc=desc, unit="batch", leave=False)
    for batch in progress:
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler_enabled):
            logits = forward_batch(model, video, aggregation, topk)
            loss = criterion(logits, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.numel()
        total_loss += loss.detach().item() * batch_size
        total_items += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_logits.extend(logits.detach().float().cpu().tolist())
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=total_loss / max(1, total_items))

    return compute_metrics(
        labels=all_labels,
        logits=all_logits,
        loss=total_loss / max(1, total_items),
        threshold=threshold,
    )


def checkpoint_payload(
    model: EchoPrimeBinaryClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    best_metric: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model.config(),
        "args": vars(args),
        "metrics": metrics,
        "best_metric": best_metric,
    }


def metric_improved(name: str, value: float, best: float) -> bool:
    if math.isnan(value):
        return False
    if name == "val_loss":
        return value < best
    return value > best


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(metrics_path: Path, output_dir: Path) -> None:
    try:
        from plot_training_curves import plot_metrics
    except Exception as exc:
        print(f"Skipping plots because plot helper could not be imported: {exc}")
        return
    try:
        plot_metrics(metrics_path, output_dir / "curves")
    except Exception as exc:
        print(f"Skipping plots because plotting failed: {exc}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = build_loaders(args)
    model = EchoPrimeBinaryClassifier(
        weights_path=args.weights_path,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    pos_weight = positive_weight(train_loader.dataset.labels(), args.pos_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    metric_rows: list[dict[str, Any]] = []
    best_value = float("inf") if args.best_metric == "val_loss" else -float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            aggregation=args.eval_aggregation,
            topk=args.topk,
            optimizer=optimizer,
            amp=args.amp,
            scaler=scaler,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d} train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                threshold=args.threshold,
                aggregation=args.eval_aggregation,
                topk=args.topk,
                optimizer=None,
                amp=args.amp,
                desc=f"Epoch {epoch:03d}/{args.epochs:03d} val",
            )
        scheduler.step(val_metrics["loss"])

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr_encoder": optimizer.param_groups[0]["lr"]
            if not args.freeze_encoder
            else 0.0,
            "lr_head": optimizer.param_groups[-1]["lr"],
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        metric_rows.append(row)
        write_metrics(output_dir / "metrics.csv", metric_rows)

        selected_value = row[args.best_metric]
        if args.best_metric == "val_auc" and math.isnan(float(selected_value)):
            selected_value = -val_metrics["loss"]

        is_best = metric_improved(args.best_metric, float(selected_value), best_value)
        if is_best:
            best_value = float(selected_value)

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            metrics=row,
            best_metric=best_value,
        )
        torch.save(payload, output_dir / "last.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

    if not args.no_plot_curves:
        maybe_plot(output_dir / "metrics.csv", output_dir)


if __name__ == "__main__":
    main()
