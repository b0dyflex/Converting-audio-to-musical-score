"""
plot_training.py
================
Визуализация метрик обучения из CSV-файлов, сохранённых train.py.

Запуск:
    python plot_training.py --metrics_dir checkpoints/
    python plot_training.py --metrics_dir checkpoints/ --save plots/training.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.rcParams.update({
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.2,
    "grid.linestyle":   "--",
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#0d0d0d",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "grid.color":       "#333333",
    "text.color":       "#cccccc",
})

ACCENT_TRAIN = "#00e5ff"   # cyan
ACCENT_VAL   = "#ff4081"   # pink
ACCENT_LR    = "#b9ff66"   # lime
ACCENT_STEP  = "#ffd740"   # amber


def load_csv(path: Path) -> dict[str, list]:
    if not path.exists():
        return {}
    data: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def plot(metrics_dir: str, save_path: str | None = None):
    base = Path(metrics_dir)
    epoch_data = load_csv(base / "epoch_metrics.csv")
    step_data  = load_csv(base / "step_metrics.csv")

    has_epoch = bool(epoch_data)
    has_step  = bool(step_data)

    if not has_epoch and not has_step:
        print(f"[ERROR] CSV-файлы не найдены в {base}")
        print("Убедитесь, что train.py завершил хотя бы одну эпоху.")
        return

    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
    fig.suptitle("SCORE GENERATION MODEL — TRAINING DASHBOARD",
                 fontsize=14, fontweight="bold", color="#ffffff",
                 y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.09)

    # ── 1. Loss по эпохам ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if has_epoch:
        epochs = epoch_data["epoch"]
        ax1.plot(epochs, epoch_data["train_loss"], color=ACCENT_TRAIN,
                 lw=2, marker="o", markersize=3, label="Train")
        ax1.plot(epochs, epoch_data["val_loss"], color=ACCENT_VAL,
                 lw=2, marker="s", markersize=3, label="Val", linestyle="--")
        best_val_idx = int(np.argmin(epoch_data["val_loss"]))
        ax1.axvline(epochs[best_val_idx], color=ACCENT_VAL, alpha=0.3, lw=1, linestyle=":")
        ax1.scatter([epochs[best_val_idx]], [epoch_data["val_loss"][best_val_idx]],
                    color=ACCENT_VAL, s=80, zorder=5, label=f"Best val={epoch_data['val_loss'][best_val_idx]:.4f}")
    ax1.set_title("LOSS / EPOCH", fontsize=10, color="#aaaaaa", pad=8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=8)

    # ── 2. Accuracy по эпохам ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if has_epoch:
        ax2.plot(epochs, [v * 100 for v in epoch_data["train_acc"]],
                 color=ACCENT_TRAIN, lw=2, marker="o", markersize=3, label="Train")
        ax2.plot(epochs, [v * 100 for v in epoch_data["val_acc"]],
                 color=ACCENT_VAL, lw=2, marker="s", markersize=3, label="Val", linestyle="--")
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
    ax2.set_title("ACCURACY / EPOCH", fontsize=10, color="#aaaaaa", pad=8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Token Accuracy")
    ax2.legend(fontsize=8)

    # ── 3. Learning Rate ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if has_step:
        steps = step_data["global_step"]
        ax3.plot(steps, step_data["lr"], color=ACCENT_LR, lw=1.2, alpha=0.9)
        ax3.fill_between(steps, step_data["lr"], alpha=0.15, color=ACCENT_LR)
    elif has_epoch:
        ax3.plot(epoch_data["epoch"], epoch_data["lr"], color=ACCENT_LR,
                 lw=2, marker="o", markersize=3)
    ax3.set_title("LEARNING RATE", fontsize=10, color="#aaaaaa", pad=8)
    ax3.set_xlabel("Step" if has_step else "Epoch")
    ax3.set_ylabel("LR")
    ax3.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # ── 4. Step Loss (сглаженный) ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    if has_step:
        steps = step_data["global_step"]
        raw   = step_data["loss"]
        sm    = smooth(raw, window=max(1, len(raw) // 50))
        ax4.plot(steps, raw, color=ACCENT_STEP, lw=0.5, alpha=0.25, label="Raw")
        ax4.plot(steps, sm,  color=ACCENT_STEP, lw=2,   alpha=0.95, label="Smoothed")
        ax4.legend(fontsize=8)
    elif has_epoch:
        ax4.plot(epoch_data["epoch"], epoch_data["train_loss"],
                 color=ACCENT_STEP, lw=2, marker="o", markersize=3)
    ax4.set_title("STEP LOSS (TRAIN)", fontsize=10, color="#aaaaaa", pad=8)
    ax4.set_xlabel("Global Step" if has_step else "Epoch")
    ax4.set_ylabel("Loss")

    # ── 5. Epoch summary table ────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    if has_epoch:
        n_show = min(10, len(epoch_data["epoch"]))
        rows = []
        for i in range(-n_show, 0):
            rows.append([
                f"{int(epoch_data['epoch'][i])}",
                f"{epoch_data['train_loss'][i]:.4f}",
                f"{epoch_data['val_loss'][i]:.4f}",
                f"{epoch_data['train_acc'][i]*100:.1f}%",
                f"{epoch_data['val_acc'][i]*100:.1f}%",
            ])
        tbl = ax5.table(
            cellText=rows,
            colLabels=["Ep", "T.Loss", "V.Loss", "T.Acc", "V.Acc"],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#1a1a1a" if r % 2 == 0 else "#111111")
            cell.set_edgecolor("#333333")
            cell.set_text_props(color="#cccccc")
            if r == 0:
                cell.set_facecolor("#222222")
                cell.set_text_props(color=ACCENT_TRAIN, fontweight="bold")
    ax5.set_title("LAST EPOCHS", fontsize=10, color="#aaaaaa", pad=8)

    # Статистика в заголовке
    if has_epoch:
        best_val = min(epoch_data["val_loss"])
        best_acc = max(epoch_data["val_acc"]) * 100
        total_ep = int(max(epoch_data["epoch"]))
        fig.text(0.5, 0.935,
                 f"epochs: {total_ep}  |  best val loss: {best_val:.4f}  |  best val acc: {best_acc:.1f}%",
                 ha="center", fontsize=9, color="#777777")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"График сохранён: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--metrics_dir", default="checkpoints",
                        help="Папка с epoch_metrics.csv и step_metrics.csv")
    parser.add_argument("--save", default=None,
                        help="Путь для сохранения PNG (если не указан — показ на экране)")
    args = parser.parse_args()
    plot(args.metrics_dir, args.save)