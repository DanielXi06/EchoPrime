from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ECHO_PRIME_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(
    3, 1, 1, 1
)
ECHO_PRIME_STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(
    3, 1, 1, 1
)


@dataclass(frozen=True)
class VideoRecord:
    video_path: Path
    label: float
    raw: dict[str, Any]


def parse_binary_label(value: str) -> float:
    normalized = str(value).strip().lower()
    positive = {"1", "true", "yes", "positive", "disease", "abnormal"}
    negative = {"0", "false", "no", "negative", "healthy", "normal"}
    if normalized in positive:
        return 1.0
    if normalized in negative:
        return 0.0
    label = float(normalized)
    if label not in (0.0, 1.0):
        raise ValueError(f"Binary label must be 0 or 1, got {value!r}")
    return label


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


def load_video_records(
    csv_path: str | Path,
    data_root: str | Path | None = None,
    path_column: str = "video_path",
    label_column: str = "label",
    split_column: str | None = None,
    split_value: str | None = None,
) -> list[VideoRecord]:
    csv_path = Path(csv_path)
    records: list[VideoRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [c for c in (path_column, label_column) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in {csv_path}")
        if split_column and split_column not in reader.fieldnames:
            raise ValueError(f"Missing split column {split_column!r} in {csv_path}")

        for row in reader:
            if split_column and split_value is not None:
                if str(row.get(split_column, "")).strip() != str(split_value):
                    continue
            records.append(
                VideoRecord(
                    video_path=_resolve_video_path(row[path_column], csv_path, data_root),
                    label=parse_binary_label(row[label_column]),
                    raw=dict(row),
                )
            )
    if not records:
        raise ValueError(f"No records loaded from {csv_path}")
    return records


def read_video_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def _pad_window(window: np.ndarray, target_frames: int) -> np.ndarray:
    if len(window) >= target_frames:
        return window[:target_frames]
    pad_count = target_frames - len(window)
    pad_frame = window[-1:]
    padding = np.repeat(pad_frame, pad_count, axis=0)
    return np.concatenate([window, padding], axis=0)


def _window_to_clip(
    frames: np.ndarray,
    start: int,
    window_frames: int,
    frame_stride: int,
) -> np.ndarray:
    end = min(start + window_frames, len(frames))
    window = _pad_window(frames[start:end], window_frames)
    return window[::frame_stride]


def _uniform_starts(num_frames: int, window_frames: int, num_clips: int) -> list[int]:
    if num_clips <= 1 or num_frames <= window_frames:
        return [0] * max(1, num_clips)
    max_start = num_frames - window_frames
    return [int(round(v)) for v in np.linspace(0, max_start, num_clips)]


def crop_and_scale(
    img: np.ndarray,
    res: tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_CUBIC,
    zoom: float = 0.1,
) -> np.ndarray:
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    return cv2.resize(img, res, interpolation=interpolation)


def preprocess_clip(
    clip: np.ndarray,
    video_size: int = 224,
    zoom: float = 0.1,
) -> torch.Tensor:
    processed = np.zeros((len(clip), video_size, video_size, 3), dtype=np.float32)
    for i, frame in enumerate(clip):
        processed[i] = crop_and_scale(
            frame,
            res=(video_size, video_size),
            interpolation=cv2.INTER_CUBIC,
            zoom=zoom,
        )
    tensor = torch.as_tensor(processed, dtype=torch.float32).permute(3, 0, 1, 2)
    tensor.sub_(ECHO_PRIME_MEAN).div_(ECHO_PRIME_STD)
    return tensor


class EchoPrimeVideoDataset(Dataset):
    """CSV-backed AVI/MP4 dataset with EchoPrime-style video preprocessing."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path | None = None,
        path_column: str = "video_path",
        label_column: str = "label",
        split_column: str | None = None,
        split_value: str | None = None,
        mode: str = "train",
        eval_clips: int = 5,
        window_frames: int = 32,
        frame_stride: int = 2,
        video_size: int = 224,
        zoom: float = 0.1,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.records = load_video_records(
            csv_path=csv_path,
            data_root=data_root,
            path_column=path_column,
            label_column=label_column,
            split_column=split_column,
            split_value=split_value,
        )
        self.mode = mode
        self.eval_clips = int(eval_clips)
        self.window_frames = int(window_frames)
        self.frame_stride = int(frame_stride)
        self.video_size = int(video_size)
        self.zoom = float(zoom)

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> list[float]:
        return [record.label for record in self.records]

    @property
    def clip_frames(self) -> int:
        return self.window_frames // self.frame_stride

    def _sample_train_clip(self, frames: np.ndarray) -> torch.Tensor:
        if len(frames) > self.window_frames:
            start = random.randint(0, len(frames) - self.window_frames)
        else:
            start = 0
        clip = _window_to_clip(frames, start, self.window_frames, self.frame_stride)
        return preprocess_clip(clip, self.video_size, self.zoom)

    def _sample_eval_clips(self, frames: np.ndarray) -> torch.Tensor:
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
        video = (
            self._sample_train_clip(frames)
            if self.mode == "train"
            else self._sample_eval_clips(frames)
        )
        return {
            "video": video,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "path": str(record.video_path),
        }
