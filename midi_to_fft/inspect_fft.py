import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # без GUI — сохраняем в файл
import matplotlib.pyplot as plt


def _save_and_open(fig, npy_path: str):
    """Сохраняет PNG рядом с .npy файлом и открывает в Windows."""
    out = os.path.abspath(npy_path).replace(".npy", "_preview.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out}")
    try:
        if sys.platform == "win32":
            os.startfile(out)
        else:
            win_path = os.popen(f'wslpath -w "{out}"').read().strip()
            if win_path:
                os.system(f'explorer.exe "{win_path}" 2>/dev/null')
            else:
                os.system(f'xdg-open "{out}" 2>/dev/null')
    except Exception:
        pass


def show(path: str, segments: list = None):
    data = np.load(path)
    print(f"Файл:        {path}")
    print(f"Размерность: {data.shape}  dtype={data.dtype}")
    print(f"Мин={data.min():.4f}  Макс={data.max():.4f}  Среднее={data.mean():.4f}")

    # ── Новый формат: (F, T) ──────────────────────────────────
    if data.ndim == 2:
        print(f"Формат: новый  F={data.shape[0]} mel-бин × T={data.shape[1]} фреймов")
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"Mel-спектрограмма: {os.path.basename(path)}")
        ax.set_xlabel("Временной фрейм")
        ax.set_ylabel("Mel-бин")
        plt.colorbar(im, ax=ax, label="Амплитуда [0,1]")
        plt.tight_layout()
        _save_and_open(fig, path)

    # ── Старый формат: (N, F, T) ──────────────────────────────
    elif data.ndim == 3:
        N = data.shape[0]
        print(f"Формат: старый  N={N} сегм × F={data.shape[1]} × T={data.shape[2]}")

        segs = segments if segments else list(range(N))
        segs = [s for s in segs if 0 <= s < N]

        fig, axes = plt.subplots(1, len(segs),
                                 figsize=(max(6, 5 * len(segs)), 4))
        if len(segs) == 1:
            axes = [axes]

        for ax, i in zip(axes, segs):
            im = ax.imshow(data[i], aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"Сегмент {i}")
            ax.set_xlabel("Фрейм")
            ax.set_ylabel("Бин")
            plt.colorbar(im, ax=ax)

        plt.suptitle(os.path.basename(path))
        plt.tight_layout()
        _save_and_open(fig, path)

    else:
        print(f"[ОШИБКА] Неожиданная размерность: {data.ndim}D")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Путь к .npy файлу")
    parser.add_argument("--segments", type=int, nargs="+",
                        help="Индексы сегментов (только для формата N,F,T)")
    args = parser.parse_args()
    show(args.path, args.segments)
