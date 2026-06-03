from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
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

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable

from echo_prime.video_classifier import EchoPrimeBinaryClassifier
from echo_prime.video_data import (
    _uniform_starts,
    get_video_frame_count,
    parse_class_names,
    preprocess_clip,
    read_video_windows_rgb,
    window_to_clip,
)
from train_video_classifier import aggregate_clip_logits


@dataclass(frozen=True)
class UnlabeledVideoRecord:
    video_path: Path
    raw: dict[str, Any]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _resolve_video_path(
    value: str,
    csv_path: Path,
    data_root: str | Path | None,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if data_root:
        return Path(data_root) / path
    return csv_path.parent / path


def load_unlabeled_records(
    csv_path: str | Path,
    data_root: str | Path | None = None,
    path_column: str = "video_path",
    split_column: str | None = None,
    split_value: str | None = None,
) -> list[UnlabeledVideoRecord]:
    csv_path = Path(csv_path)
    records: list[UnlabeledVideoRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if path_column not in reader.fieldnames:
            raise ValueError(f"Missing required column {path_column!r} in {csv_path}")
        if split_column and split_column not in reader.fieldnames:
            raise ValueError(f"Missing split column {split_column!r} in {csv_path}")

        for row in reader:
            if split_column and split_value is not None:
                if str(row.get(split_column, "")).strip() != str(split_value):
                    continue
            records.append(
                UnlabeledVideoRecord(
                    video_path=_resolve_video_path(row[path_column], csv_path, data_root),
                    raw=dict(row),
                )
            )
    if not records:
        raise ValueError(f"No records loaded from {csv_path}")
    return records


class UnlabeledEchoPrimeVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path | None = None,
        path_column: str = "video_path",
        split_column: str | None = None,
        split_value: str | None = None,
        eval_clips: int = 5,
        window_frames: int = 32,
        frame_stride: int = 2,
        video_size: int = 224,
        zoom: float = 0.1,
    ) -> None:
        self.records = load_unlabeled_records(
            csv_path=csv_path,
            data_root=data_root,
            path_column=path_column,
            split_column=split_column,
            split_value=split_value,
        )
        self.eval_clips = int(eval_clips)
        self.window_frames = int(window_frames)
        self.frame_stride = int(frame_stride)
        self.video_size = int(video_size)
        self.zoom = float(zoom)

    def __len__(self) -> int:
        return len(self.records)

    def _sample_eval_clips(self, path: Path, num_frames: int):
        starts = _uniform_starts(num_frames, self.window_frames, self.eval_clips)
        windows = read_video_windows_rgb(path, starts, self.window_frames)
        clips = [
            preprocess_clip(
                window_to_clip(window, self.frame_stride),
                self.video_size,
                self.zoom,
            )
            for window in windows
        ]
        return torch.stack(clips, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            num_frames = get_video_frame_count(record.video_path)
            video = self._sample_eval_clips(record.video_path, num_frames)
        except Exception as exc:
            return {
                "video": None,
                "path": str(record.video_path),
                "error": str(exc),
            }
        return {
            "video": video,
            "path": str(record.video_path),
            "error": "",
        }


def collate_predict_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    valid_items = [item for item in batch if item["video"] is not None]
    skipped_items = [item for item in batch if item["video"] is None]
    videos = torch.stack([item["video"] for item in valid_items], dim=0) if valid_items else None
    return {
        "video": videos,
        "path": [item["path"] for item in valid_items],
        "skipped": [
            {
                "video_path": item["path"],
                "error": item["error"],
            }
            for item in skipped_items
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unlabeled video-level prediction with an EchoPrime classifier checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predict-csv", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--path-column", default="video_path")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--split-value", default=None)
    parser.add_argument("--task", choices=["binary", "multiclass"], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--class-names",
        default=None,
        help="Comma-separated class names. Defaults to names stored in checkpoint.",
    )
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


def resolve_task_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
) -> tuple[str, int, list[str]]:
    train_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    inferred_num_classes = int(
        model_config.get("num_classes", train_args.get("num_classes", 1))
    )
    inferred_task = train_args.get(
        "task",
        "multiclass" if inferred_num_classes > 1 else "binary",
    )

    task = args.task or inferred_task
    if task == "binary":
        num_classes = 1
    else:
        num_classes = int(args.num_classes or inferred_num_classes)
        if num_classes <= 1:
            raise ValueError("--task multiclass requires --num-classes greater than 1.")

    class_names = parse_class_names(args.class_names or train_args.get("class_names"))
    if task == "multiclass" and not class_names:
        class_names = [f"Class_{index}" for index in range(num_classes)]
    if task == "multiclass" and len(class_names) != num_classes:
        raise ValueError("--class-names length must match --num-classes.")
    return task, num_classes, class_names


def load_model(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
) -> EchoPrimeBinaryClassifier:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("model_config", {}))
    if args.weights_path:
        config["weights_path"] = args.weights_path
    if "weights_path" not in config:
        raise ValueError("Checkpoint has no model_config.weights_path; pass --weights-path.")
    config["freeze_encoder"] = True
    config["num_classes"] = num_classes
    model = EchoPrimeBinaryClassifier(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def forward_predict_batch(
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
    if flat_logits.ndim == 1:
        clip_logits = flat_logits.reshape(batch_size, num_clips)
    else:
        clip_logits = flat_logits.reshape(batch_size, num_clips, -1)
    return aggregate_clip_logits(clip_logits, aggregation, topk)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})
    task, num_classes, class_names = resolve_task_config(args, checkpoint)
    aggregation = args.eval_aggregation or train_args.get("eval_aggregation", "mean")
    topk = args.topk or int(train_args.get("topk", 3))

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / "predict"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, device, num_classes)
    dataset = UnlabeledEchoPrimeVideoDataset(
        csv_path=args.predict_csv,
        data_root=args.data_root,
        path_column=args.path_column,
        split_column=args.split_column,
        split_value=args.split_value,
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
        collate_fn=collate_predict_batch,
    )

    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        progress = tqdm(loader, total=len(loader), desc="Predicting", unit="batch")
        for batch in progress:
            skipped_rows.extend(batch["skipped"])
            if batch["video"] is None:
                continue

            video = batch["video"].to(device, non_blocking=True)
            logits = forward_predict_batch(model, video, aggregation, topk)
            if task == "binary":
                for path, logit in zip(batch["path"], logits.detach().float().cpu().tolist()):
                    probability = sigmoid(float(logit))
                    rows.append(
                        {
                            "video_path": path,
                            "logit": logit,
                            "probability": probability,
                            "prediction": 1 if probability >= args.threshold else 0,
                        }
                    )
            else:
                logits_cpu = logits.detach().float().cpu()
                probs = torch.softmax(logits_cpu, dim=1)
                for path, logit_values, prob_values in zip(
                    batch["path"],
                    logits_cpu.tolist(),
                    probs.tolist(),
                ):
                    prediction = int(torch.as_tensor(logit_values).argmax().item())
                    row: dict[str, Any] = {
                        "video_path": path,
                        "prediction": prediction,
                        "prediction_name": class_names[prediction],
                    }
                    for index, name in enumerate(class_names):
                        row[f"logit_{name}"] = logit_values[index]
                        row[f"prob_{name}"] = prob_values[index]
                    rows.append(row)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(predicted=len(rows), skipped=len(skipped_rows))

    output_path = output_dir / "predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        if task == "binary":
            fieldnames = ["video_path", "logit", "probability", "prediction"]
        else:
            fieldnames = ["video_path", "prediction", "prediction_name"]
            for name in class_names:
                fieldnames.extend([f"logit_{name}", f"prob_{name}"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    skipped_path = output_dir / "skipped_videos.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "error"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"Saved predictions to: {output_path}")
    print(f"Saved skipped videos to: {skipped_path}")
    print(
        f"num_predicted={len(rows)} num_skipped={len(skipped_rows)} "
        f"task={task} threshold={args.threshold} aggregation={aggregation}"
    )


if __name__ == "__main__":
    main()
