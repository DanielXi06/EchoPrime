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

from echo_prime.video_classifier import EchoPrimeBinaryClassifier
from echo_prime.video_data import (
    _uniform_starts,
    _window_to_clip,
    preprocess_clip,
    read_video_rgb,
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

    def _sample_eval_clips(self, frames):
        starts = _uniform_starts(len(frames), self.window_frames, self.eval_clips)
        clips = [
            preprocess_clip(
                _window_to_clip(frames, start, self.window_frames, self.frame_stride),
                self.video_size,
                self.zoom,
            )
            for start in starts
        ]
        return torch.stack(clips, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = read_video_rgb(record.video_path)
        return {
            "video": self._sample_eval_clips(frames),
            "path": str(record.video_path),
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


def load_model(args: argparse.Namespace, device: torch.device) -> EchoPrimeBinaryClassifier:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("model_config", {}))
    if args.weights_path:
        config["weights_path"] = args.weights_path
    if "weights_path" not in config:
        raise ValueError("Checkpoint has no model_config.weights_path; pass --weights-path.")
    config["freeze_encoder"] = True
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
        else Path(args.checkpoint).resolve().parent / "predict"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
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
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            logits = forward_predict_batch(model, video, aggregation, topk)
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

    output_path = output_dir / "predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "logit", "probability", "prediction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {output_path}")
    print(f"num_samples={len(rows)} threshold={args.threshold} aggregation={aggregation}")


if __name__ == "__main__":
    main()
