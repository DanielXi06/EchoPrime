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
    label: float | int
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


def parse_class_names(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_label(
    value: str,
    num_classes: int = 1,
    class_names: str | list[str] | tuple[str, ...] | None = None,
) -> float | int:
    if int(num_classes) == 1:
        return parse_binary_label(value)

    normalized = str(value).strip()
    names = parse_class_names(class_names)
    if names:
        lookup = {name.lower(): index for index, name in enumerate(names)}
        matched = lookup.get(normalized.lower())
        if matched is not None:
            return matched

    try:
        numeric = float(normalized)
    except ValueError as exc:
        if names:
            raise ValueError(
                f"Unknown class label {value!r}; expected one of {names} "
                "or an integer class id."
            ) from exc
        raise ValueError(f"Class label must be an integer id, got {value!r}") from exc

    if not numeric.is_integer():
        raise ValueError(f"Class label must be an integer id, got {value!r}")
    label = int(numeric)
    if label < 0 or label >= int(num_classes):
        raise ValueError(
            f"Class label must be in [0, {int(num_classes) - 1}], got {value!r}"
        )
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
    num_classes: int = 1,
    class_names: str | list[str] | tuple[str, ...] | None = None,
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
                    label=parse_label(
                        row[label_column],
                        num_classes=num_classes,
                        class_names=class_names,
                    ),
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


def get_video_frame_count(path: str | Path) -> int:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine frame count for video: {path}")
    return frame_count


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _pad_window(window: np.ndarray, target_frames: int) -> np.ndarray:
    if len(window) >= target_frames:
        return window[:target_frames]
    pad_count = target_frames - len(window)
    pad_frame = window[-1:]
    padding = np.repeat(pad_frame, pad_count, axis=0)
    return np.concatenate([window, padding], axis=0)


def _read_window_from_capture(
    cap: cv2.VideoCapture,
    start: int,
    window_frames: int,
) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))

    frames: list[np.ndarray] = []
    for _ in range(window_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_frame_to_rgb(frame))

    if not frames:
        raise RuntimeError(f"No frames decoded from requested window starting at {start}")
    return _pad_window(np.stack(frames, axis=0), window_frames)


def read_video_windows_rgb(
    path: str | Path,
    starts: list[int],
    window_frames: int,
) -> list[np.ndarray]:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        return [
            _read_window_from_capture(cap, start=max(0, int(start)), window_frames=window_frames)
            for start in starts
        ]
    finally:
        cap.release()


def window_to_clip(window: np.ndarray, frame_stride: int) -> np.ndarray:
    return window[::frame_stride]


def sample_train_start(num_frames: int, window_frames: int) -> int:
    max_start = max(0, int(num_frames) - int(window_frames))
    if max_start == 0:
        return 0
    return random.randint(0, max_start)


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
        num_classes: int = 1,
        class_names: str | list[str] | tuple[str, ...] | None = None,
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
            num_classes=num_classes,
            class_names=class_names,
        )
        self.mode = mode
        self.eval_clips = int(eval_clips)
        self.window_frames = int(window_frames)
        self.frame_stride = int(frame_stride)
        self.video_size = int(video_size)
        self.zoom = float(zoom)
        self.num_classes = int(num_classes)
        self.class_names = parse_class_names(class_names)

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> list[float | int]:
        return [record.label for record in self.records]

    @property
    def clip_frames(self) -> int:
        return self.window_frames // self.frame_stride

    def _sample_train_clip(self, path: Path, num_frames: int) -> torch.Tensor:
        start = sample_train_start(num_frames, self.window_frames)
        window = read_video_windows_rgb(path, [start], self.window_frames)[0]
        clip = window_to_clip(window, self.frame_stride)
        return preprocess_clip(clip, self.video_size, self.zoom)

    def _sample_eval_clips(self, path: Path, num_frames: int) -> torch.Tensor:
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
        num_frames = get_video_frame_count(record.video_path)
        video = (
            self._sample_train_clip(record.video_path, num_frames)
            if self.mode == "train"
            else self._sample_eval_clips(record.video_path, num_frames)
        )
        return {
            "video": video,
            "label": torch.tensor(
                record.label,
                dtype=torch.float32 if self.num_classes == 1 else torch.long,
            ),
            "path": str(record.video_path),
        }
