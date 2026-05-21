from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from echo_prime.video_classifier import EchoPrimeBinaryClassifier
from echo_prime.video_data import EchoPrimeVideoDataset
from train_video_classifier import aggregate_clip_logits, binary_auc, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an EchoPrime video-level binary classifier checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--path-column", default="video_path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Override encoder weights path used only to initialize the model skeleton.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-clips", type=int, default=5)
    parser.add_argument(
        "--eval-aggregation",
        choices=["mean", "max", "topk_mean"],
        default=None,
    )
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Physical GPU id to expose before torch imports, e.g. --gpu-id 1.",
    )
    return parser.parse_args()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def load_model(args: argparse.Namespace, device: torch.device) -> EchoPrimeBinaryClassifier:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("model_config", {}))
    if args.weights_path:
        config["weights_path"] = args.weights_path
    if "weights_path" not in config:
        raise ValueError("Checkpoint has no model_config.weights_path; pass --weights-path.")
    config["freeze_encoder"] = False
    model = EchoPrimeBinaryClassifier(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def forward_eval_batch(
    model: EchoPrimeBinaryClassifier,
    video: torch.Tensor,
    aggregation: str,
    topk: int,
) -> torch.Tensor:
    if video.ndim != 6:
        raise ValueError(f"Expected eval video tensor B x K x C x T x H x W, got {video.shape}")
    batch_size, num_clips, channels, frames, height, width = video.shape
    flat_video = video.reshape(batch_size * num_clips, channels, frames, height, width)
    flat_logits = model(flat_video)
    clip_logits = flat_logits.reshape(batch_size, num_clips)
    return aggregate_clip_logits(clip_logits, aggregation, topk)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})
    aggregation = args.eval_aggregation or train_args.get("eval_aggregation", "mean")
    topk = args.topk or int(train_args.get("topk", 3))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / "test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    dataset = EchoPrimeVideoDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        path_column=args.path_column,
        label_column=args.label_column,
        split_column=args.split_column,
        split_value=args.test_split,
        mode="eval",
        eval_clips=args.eval_clips,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    criterion = nn.BCEWithLogitsLoss()

    all_labels: list[float] = []
    all_logits: list[float] = []
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            logits = forward_eval_batch(model, video, aggregation, topk)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.numel()
            total_items += labels.numel()

            batch_logits = logits.detach().float().cpu().tolist()
            batch_labels = labels.detach().cpu().tolist()
            all_logits.extend(batch_logits)
            all_labels.extend(batch_labels)
            for path, label, logit in zip(batch["path"], batch_labels, batch_logits):
                prob = sigmoid(float(logit))
                rows.append(
                    {
                        "video_path": path,
                        "label": label,
                        "logit": logit,
                        "probability": prob,
                        "prediction": 1 if prob >= args.threshold else 0,
                    }
                )

    metrics = compute_metrics(
        labels=all_labels,
        logits=all_logits,
        loss=total_loss / max(1, total_items),
        threshold=args.threshold,
    )
    metrics["num_samples"] = len(all_labels)
    metrics["threshold"] = args.threshold
    metrics["eval_clips"] = args.eval_clips
    metrics["eval_aggregation"] = aggregation
    metrics["topk"] = topk
    metrics["auc"] = binary_auc(all_labels, [sigmoid(v) for v in all_logits])

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "label", "logit", "probability", "prediction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
