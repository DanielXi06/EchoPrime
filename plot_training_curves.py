from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_metrics(metrics_csv: str | Path) -> dict[str, list[float]]:
    metrics_csv = Path(metrics_csv)
    columns: dict[str, list[float]] = {}
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, [])
                try:
                    columns[key].append(float(value))
                except (TypeError, ValueError):
                    columns[key].append(float("nan"))
    return columns


def _plot_pair(columns: dict[str, list[float]], keys: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = columns.get("epoch")
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    plotted = False
    for key in keys:
        if key in columns:
            plt.plot(epochs, columns[key], marker="o", label=key)
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_metrics(metrics_csv: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = _read_metrics(metrics_csv)
    groups = {
        "loss.png": ["train_loss", "val_loss"],
        "accuracy.png": ["train_acc", "val_acc"],
        "auc.png": ["train_auc", "val_auc"],
        "f1.png": ["train_f1", "val_f1"],
        "sensitivity.png": ["train_sensitivity", "val_sensitivity"],
        "specificity.png": ["train_specificity", "val_specificity"],
        "learning_rate.png": ["lr_encoder", "lr_head"],
    }
    for filename, keys in groups.items():
        _plot_pair(columns, keys, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.metrics_csv).resolve().parent / "curves"
    )
    plot_metrics(args.metrics_csv, output_dir)


if __name__ == "__main__":
    main()
