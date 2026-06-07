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
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable

from echo_prime.video_classifier import EchoPrimeBinaryClassifier
from echo_prime.video_data import EchoPrimeVideoDataset, parse_class_names
from train_video_classifier import aggregate_clip_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cascade binary disease screening and three-class disease typing."
    )
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--threeclass-checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--path-column", default="video_path")
    parser.add_argument("--label-column", default="class_name")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Override encoder weights path used only to initialize model skeletons.",
    )
    parser.add_argument(
        "--final-class-names",
        default="healthy,ASD,VSD,PDA",
        help="Final four-class label order for metrics and confusion matrix.",
    )
    parser.add_argument(
        "--disease-class-names",
        default="ASD,VSD,PDA",
        help="Three-class disease label order used by the second-stage model.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-clips", type=int, default=5)
    parser.add_argument(
        "--eval-aggregation",
        choices=["mean", "max", "topk_mean"],
        default="mean",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="Disease probability threshold for sending a video to the three-class model.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Physical GPU id to expose before torch imports, e.g. --gpu-id 1.",
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    weights_path: str | Path | None,
    num_classes: int,
) -> EchoPrimeBinaryClassifier:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("model_config", {}))
    if weights_path:
        config["weights_path"] = str(weights_path)
    if "weights_path" not in config:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no model_config.weights_path; "
            "pass --weights-path."
        )
    config["freeze_encoder"] = False
    config["num_classes"] = int(num_classes)
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
        raise ValueError(f"Expected B x K x C x T x H x W, got {tuple(video.shape)}")
    batch_size, num_clips, channels, frames, height, width = video.shape
    flat_video = video.reshape(batch_size * num_clips, channels, frames, height, width)
    flat_logits = model(flat_video)
    if flat_logits.ndim == 1:
        clip_logits = flat_logits.reshape(batch_size, num_clips)
    else:
        clip_logits = flat_logits.reshape(batch_size, num_clips, -1)
    return aggregate_clip_logits(clip_logits, aggregation, topk)


def build_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        if 0 <= label < num_classes and 0 <= prediction < num_classes:
            matrix[label, prediction] += 1
    return matrix


def compute_multiclass_metrics(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> dict[str, float]:
    if not labels:
        return {
            "acc": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
        }

    matrix = build_confusion_matrix(labels, predictions, num_classes)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_index in range(num_classes):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - matrix[class_index, class_index])
        fn = float(matrix[class_index, :].sum() - matrix[class_index, class_index])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(predictions, dtype=np.int64)
    return {
        "acc": float((labels_arr == preds_arr).mean()),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
    }


def save_confusion_outputs(
    labels: list[int],
    predictions: list[int],
    label_order: list[str],
    output_dir: Path,
    tag: str,
    threshold: float,
) -> None:
    matrix = build_confusion_matrix(labels, predictions, len(label_order))
    row_sums = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    with (output_dir / f"confusion_matrix_counts_{tag}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *label_order])
        for label_name, row in zip(label_order, matrix):
            writer.writerow([label_name, *[int(value) for value in row]])

    with (output_dir / f"confusion_matrix_percent_{tag}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *label_order])
        for label_name, row in zip(label_order, percent):
            writer.writerow([label_name, *[f"{value:.4f}" for value in row]])

    payload = {
        "label_order": label_order,
        "counts": matrix.tolist(),
        "row_percent": percent.tolist(),
        "binary_threshold": threshold,
    }
    (output_dir / f"confusion_matrix_{tag}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    image = ax.imshow(percent, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(label_order)), labels=label_order)
    ax.set_yticks(range(len(label_order)), labels=label_order)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Cascade Video-Level Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{percent[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Percentage within true class (%)")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{tag}.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    final_class_names = parse_class_names(args.final_class_names)
    disease_class_names = parse_class_names(args.disease_class_names)
    if len(final_class_names) != 4:
        raise ValueError("--final-class-names must contain exactly four classes.")
    if len(disease_class_names) != 3:
        raise ValueError("--disease-class-names must contain exactly three classes.")

    final_lookup = {name.lower(): index for index, name in enumerate(final_class_names)}
    disease_to_final = []
    for disease_name in disease_class_names:
        final_index = final_lookup.get(disease_name.lower())
        if final_index is None:
            raise ValueError(
                f"Disease class {disease_name!r} is absent from --final-class-names."
            )
        disease_to_final.append(final_index)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.binary_checkpoint).resolve().parent / "cascade_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cascade_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    binary_model = load_checkpoint_model(
        checkpoint_path=args.binary_checkpoint,
        device=device,
        weights_path=args.weights_path,
        num_classes=1,
    )
    threeclass_model = load_checkpoint_model(
        checkpoint_path=args.threeclass_checkpoint,
        device=device,
        weights_path=args.weights_path,
        num_classes=3,
    )

    dataset = EchoPrimeVideoDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        path_column=args.path_column,
        label_column=args.label_column,
        split_column=args.split_column,
        split_value=args.test_split,
        mode="eval",
        eval_clips=args.eval_clips,
        num_classes=len(final_class_names),
        class_names=final_class_names,
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
    all_labels: list[int] = []
    all_predictions: list[int] = []
    gate_positive_count = 0
    gate_negative_count = 0
    amp_enabled = args.amp and device.type == "cuda"

    with torch.no_grad():
        progress = tqdm(loader, total=len(loader), desc="Cascade testing", unit="batch")
        for batch in progress:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].long()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                binary_logits = forward_eval_batch(
                    binary_model,
                    video,
                    args.eval_aggregation,
                    args.topk,
                )
            binary_logits_cpu = binary_logits.detach().float().cpu()
            binary_probs = torch.sigmoid(binary_logits_cpu)
            binary_predictions = (binary_probs >= args.binary_threshold).long()

            final_predictions = torch.zeros_like(binary_predictions)
            disease_indices = torch.nonzero(binary_predictions == 1, as_tuple=False).flatten()

            three_logits_by_batch_index: dict[int, list[float]] = {}
            three_probs_by_batch_index: dict[int, list[float]] = {}
            three_pred_by_batch_index: dict[int, int] = {}

            if disease_indices.numel() > 0:
                disease_video = video.index_select(
                    0,
                    disease_indices.to(device, non_blocking=True),
                )
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    three_logits = forward_eval_batch(
                        threeclass_model,
                        disease_video,
                        args.eval_aggregation,
                        args.topk,
                    )
                three_logits_cpu = three_logits.detach().float().cpu()
                three_probs_cpu = torch.softmax(three_logits_cpu, dim=1)
                three_predictions = torch.argmax(three_logits_cpu, dim=1)

                for local_index, batch_index_tensor in enumerate(disease_indices):
                    batch_index = int(batch_index_tensor.item())
                    disease_prediction = int(three_predictions[local_index].item())
                    final_predictions[batch_index] = disease_to_final[disease_prediction]
                    three_logits_by_batch_index[batch_index] = three_logits_cpu[
                        local_index
                    ].tolist()
                    three_probs_by_batch_index[batch_index] = three_probs_cpu[
                        local_index
                    ].tolist()
                    three_pred_by_batch_index[batch_index] = disease_prediction

            batch_labels = labels.tolist()
            batch_final_predictions = final_predictions.tolist()
            all_labels.extend(int(label) for label in batch_labels)
            all_predictions.extend(int(prediction) for prediction in batch_final_predictions)
            gate_positive_count += int((binary_predictions == 1).sum().item())
            gate_negative_count += int((binary_predictions == 0).sum().item())

            for batch_index, path in enumerate(batch["path"]):
                label = int(batch_labels[batch_index])
                final_prediction = int(batch_final_predictions[batch_index])
                binary_prediction = int(binary_predictions[batch_index].item())
                row: dict[str, Any] = {
                    "video_path": path,
                    "label": label,
                    "label_name": final_class_names[label],
                    "binary_logit": float(binary_logits_cpu[batch_index].item()),
                    "binary_probability": float(binary_probs[batch_index].item()),
                    "binary_prediction": binary_prediction,
                    "binary_prediction_name": "Disease"
                    if binary_prediction == 1
                    else "healthy",
                    "final_prediction": final_prediction,
                    "final_prediction_name": final_class_names[final_prediction],
                    "threeclass_prediction": "",
                    "threeclass_prediction_name": "",
                }
                if batch_index in three_pred_by_batch_index:
                    disease_prediction = three_pred_by_batch_index[batch_index]
                    row["threeclass_prediction"] = disease_prediction
                    row["threeclass_prediction_name"] = disease_class_names[disease_prediction]
                    for disease_index, disease_name in enumerate(disease_class_names):
                        row[f"threeclass_logit_{disease_name}"] = (
                            three_logits_by_batch_index[batch_index][disease_index]
                        )
                        row[f"threeclass_prob_{disease_name}"] = (
                            three_probs_by_batch_index[batch_index][disease_index]
                        )
                else:
                    for disease_name in disease_class_names:
                        row[f"threeclass_logit_{disease_name}"] = ""
                        row[f"threeclass_prob_{disease_name}"] = ""
                rows.append(row)

            if hasattr(progress, "set_postfix"):
                metrics_so_far = compute_multiclass_metrics(
                    all_labels,
                    all_predictions,
                    len(final_class_names),
                )
                progress.set_postfix(
                    acc=metrics_so_far["acc"],
                    gate_pos=gate_positive_count,
                )

    metrics = compute_multiclass_metrics(
        all_labels,
        all_predictions,
        len(final_class_names),
    )
    metrics.update(
        {
            "num_samples": len(all_labels),
            "binary_threshold": args.binary_threshold,
            "gate_positive_count": gate_positive_count,
            "gate_negative_count": gate_negative_count,
            "eval_clips": args.eval_clips,
            "eval_aggregation": args.eval_aggregation,
            "topk": args.topk,
            "final_class_names": final_class_names,
            "disease_class_names": disease_class_names,
        }
    )

    tag = f"cascade_threshold_{args.binary_threshold:.3f}".replace(".", "p")
    save_confusion_outputs(
        labels=all_labels,
        predictions=all_predictions,
        label_order=final_class_names,
        output_dir=output_dir,
        tag=tag,
        threshold=args.binary_threshold,
    )

    with (output_dir / "cascade_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fieldnames = [
        "video_path",
        "label",
        "label_name",
        "binary_logit",
        "binary_probability",
        "binary_prediction",
        "binary_prediction_name",
        "threeclass_prediction",
        "threeclass_prediction_name",
    ]
    for disease_name in disease_class_names:
        fieldnames.extend([f"threeclass_logit_{disease_name}", f"threeclass_prob_{disease_name}"])
    fieldnames.extend(["final_prediction", "final_prediction_name"])
    with (output_dir / "cascade_predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Cascade predictions saved to: {output_dir / 'cascade_predictions.csv'}")
    print(f"Cascade confusion matrix saved to: {output_dir}")


if __name__ == "__main__":
    main()
