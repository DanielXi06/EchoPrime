from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL_ORDER = ["Disease", "Healthy"]
PREDICTION_TO_LABEL = {1: "Disease", 0: "Healthy"}
ASD_STATS_DATASET_MARKER = "(56例统计学、24例阳性)"
VSD_POSITIVE_SEGMENT_RE = re.compile(r"(?:^|_)\d+-1(?=_|$)")


def is_asd_positive_path(path_text: str) -> bool:
    stem = Path(path_text).stem
    text = f"{path_text} {stem}"
    if ASD_STATS_DATASET_MARKER in text:
        suffix_after_marker = text.split(ASD_STATS_DATASET_MARKER, 1)[1]
        return "阳性" in suffix_after_marker
    return "阳性" in text or "positive" in text.lower()


def is_vsd_positive_path(path_text: str) -> bool:
    stem = Path(path_text).stem
    return VSD_POSITIVE_SEGMENT_RE.search(stem) is not None


def is_pda_positive_path(path_text: str) -> bool:
    positive_markers = (
        "PDA数据集",
        "PDA鏁版嵁",
        "PDA",
    )
    return any(marker in path_text for marker in positive_markers)


def infer_disease_type(path_text: str) -> str | None:
    upper_text = path_text.upper()
    if "ASD" in upper_text:
        return "ASD"
    if "VSD" in upper_text:
        return "VSD"
    if "PDA" in upper_text or "动脉导管" in path_text or "鍔ㄨ剦瀵肩" in path_text:
        return "PDA"
    return None


def has_healthy_marker(path_text: str) -> bool:
    lower_text = path_text.lower()
    healthy_markers = (
        "healthy",
        "normal",
        "negative",
        "control",
        "阴性",
        "正常",
        "健康",
    )
    return any(marker in lower_text or marker in path_text for marker in healthy_markers)


def infer_true_label(path_text: str, unknown_policy: str) -> str | None:
    disease = infer_disease_type(path_text)
    if disease == "ASD":
        return "Disease" if is_asd_positive_path(path_text) else "Healthy"
    if disease == "VSD":
        return "Disease" if is_vsd_positive_path(path_text) else "Healthy"
    if disease == "PDA":
        return "Disease" if is_pda_positive_path(path_text) else "Healthy"

    if "阳性" in path_text or "positive" in path_text.lower():
        return "Disease"
    if has_healthy_marker(path_text):
        return "Healthy"

    if unknown_policy == "healthy":
        return "Healthy"
    if unknown_policy == "skip":
        return None
    raise ValueError(
        "Could not infer true label from path. Add a path marker or use "
        f"--unknown-policy skip/healthy. Path: {path_text}"
    )


def predicted_label_from_row(
    row: dict[str, str],
    threshold: float | None,
    probability_column: str,
    prediction_column: str,
) -> str:
    if threshold is not None:
        probability = float(row[probability_column])
        return "Disease" if probability >= threshold else "Healthy"
    prediction = int(float(row[prediction_column]))
    if prediction not in PREDICTION_TO_LABEL:
        raise ValueError(f"Prediction must be 0 or 1, got {prediction}")
    return PREDICTION_TO_LABEL[prediction]


def load_rows(
    predictions_csv: str | Path,
    path_column: str,
    probability_column: str,
    prediction_column: str,
    threshold: float | None,
    unknown_policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(predictions_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {predictions_csv}")
        required = [path_column]
        required.append(probability_column if threshold is not None else prediction_column)
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in {predictions_csv}")

        for row in reader:
            path_text = row[path_column]
            true_label = infer_true_label(path_text, unknown_policy)
            if true_label is None:
                continue
            pred_label = predicted_label_from_row(
                row,
                threshold=threshold,
                probability_column=probability_column,
                prediction_column=prediction_column,
            )
            rows.append(
                {
                    "video_path": path_text,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "probability": row.get(probability_column, ""),
                    "prediction": row.get(prediction_column, ""),
                }
            )
    if not rows:
        raise ValueError("No rows available after label inference.")
    return rows


def build_confusion_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    matrix = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)
    for row in rows:
        i = label_to_index[row["true_label"]]
        j = label_to_index[row["predicted_label"]]
        matrix[i, j] += 1
    return matrix


def matrix_to_percent(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100.0


def compute_binary_metrics(matrix: np.ndarray) -> dict[str, float]:
    disease_idx = LABEL_ORDER.index("Disease")
    healthy_idx = LABEL_ORDER.index("Healthy")
    tp = int(matrix[disease_idx, disease_idx])
    fn = int(matrix[disease_idx, healthy_idx])
    fp = int(matrix[healthy_idx, disease_idx])
    tn = int(matrix[healthy_idx, healthy_idx])
    total = tp + fn + fp + tn
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "total": total,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn)
            else float("nan")
        ),
    }


def save_tables(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    percent: np.ndarray,
    output_dir: Path,
    tag: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"confusion_matrix_binary_counts_{tag}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *LABEL_ORDER])
        for label, values in zip(LABEL_ORDER, matrix):
            writer.writerow([label, *[int(v) for v in values]])

    with (output_dir / f"confusion_matrix_binary_percent_{tag}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *LABEL_ORDER])
        for label, values in zip(LABEL_ORDER, percent):
            writer.writerow([label, *[f"{v:.4f}" for v in values]])

    with (output_dir / f"predictions_with_inferred_labels_{tag}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "true_label", "predicted_label", "probability", "prediction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "label_order": LABEL_ORDER,
        "counts": matrix.tolist(),
        "row_percent": percent.tolist(),
        "metrics": compute_binary_metrics(matrix),
    }
    (output_dir / f"confusion_matrix_binary_{tag}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def plot_confusion_matrix(matrix: np.ndarray, percent: np.ndarray, output_path: Path, title: str) -> None:
    annotations = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annotations[i, j] = f"{matrix[i, j]}\n{percent[i, j]:.1f}%"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    im = ax.imshow(percent, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), labels=LABEL_ORDER)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, annotations[i, j], ha="center", va="center", color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage within true class (%)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a binary confusion matrix from EchoPrime predictions and path-derived labels."
    )
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--path-column", default="video_path")
    parser.add_argument("--probability-column", default="probability")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, recompute predicted label from probability >= threshold. Otherwise use prediction column.",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=["error", "skip", "healthy"],
        default="error",
        help="What to do when true label cannot be inferred from video_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_csv.parent / "confusion"
    tag = (
        f"threshold_{args.threshold:.3f}".replace(".", "p")
        if args.threshold is not None
        else "saved_prediction"
    )

    rows = load_rows(
        predictions_csv=predictions_csv,
        path_column=args.path_column,
        probability_column=args.probability_column,
        prediction_column=args.prediction_column,
        threshold=args.threshold,
        unknown_policy=args.unknown_policy,
    )
    matrix = build_confusion_matrix(rows)
    percent = matrix_to_percent(matrix)
    save_tables(rows, matrix, percent, output_dir, tag)
    plot_confusion_matrix(
        matrix,
        percent,
        output_dir / f"confusion_matrix_binary_{tag}.png",
        title=f"Binary Video-Level Confusion Matrix\nCell = Count + Row Percentage ({tag})",
    )

    print("Binary video-level confusion counts:")
    for label, values in zip(LABEL_ORDER, matrix):
        print(label, dict(zip(LABEL_ORDER, [int(v) for v in values])))
    print(f"Saved confusion matrix outputs to: {output_dir}")


if __name__ == "__main__":
    main()
