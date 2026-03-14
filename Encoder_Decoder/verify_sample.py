"""
verify_sample.py — ручная проверка одного сэмпла датасета
==========================================================
Показывает для конкретной папки sample_XXXXX/:
  1. Спектрограмму (текстово и если есть matplotlib — графически)
  2. Список токенов с расшифровкой
  3. Информацию из meta.json (какой MIDI, какой сегмент, сколько нот)

Это позволяет вручную убедиться, что спектрограмма и токены
соответствуют одному и тому же временному окну.

Запуск:
    # Проверить конкретную папку
    python verify_sample.py --sample_dir dataset/sample_00042_003

    # Проверить случайные 5 сэмплов из датасета
    python verify_sample.py --dataset_dir dataset --random 5

    # Сохранить PNG-картинку спектрограммы
    python verify_sample.py --sample_dir dataset/sample_00042_003 --save_png
"""

import argparse
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "midi_to_fft"))

import numpy as np

from tokenizer import decode_token, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ──────────────────────────────────────────────────────────────
def check_one(sample_dir: Path, save_png: bool = False) -> None:
    sample_dir = Path(sample_dir)

    # ── 1. Загружаем файлы ────────────────────────────────────
    spec_path = sample_dir / "spectrogram.npy"
    tokens_path = sample_dir / "tokens.npy"
    meta_path = sample_dir / "meta.json"

    if not spec_path.exists() or not tokens_path.exists():
        print(f"[ОШИБКА] Не найдены spectrogram.npy / tokens.npy в {sample_dir}")
        return

    spec = np.load(spec_path)  # (F, T) float32
    tokens = np.load(tokens_path)  # (max_seq_len,) int64

    # ── 2. Мета-информация ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Сэмпл: {sample_dir.name}")
    print(f"{'=' * 60}")

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"MIDI-файл:  {meta.get('midi_file', '?')}")
        print(f"Сегмент:    {meta.get('segment_idx', '?')}  "
              f"[{meta.get('start_sec', '?')} – {meta.get('end_sec', '?')} сек]")
        print(f"Нот в окне: {meta.get('n_notes', '?')}")
        print(f"Токенов:    {meta.get('n_tokens', '?')} (без PAD/BOS/EOS)")
    else:
        print("[INFO] meta.json не найден")

    # ── 3. Спектрограмма ──────────────────────────────────────
    print(f"\n--- Спектрограмма ---")
    print(f"Форма:  {spec.shape}  (mel-бины × временные фреймы)")
    print(f"Мин:    {spec.min():.4f}")
    print(f"Макс:   {spec.max():.4f}")
    print(f"Среднее:{spec.mean():.4f}")
    print(f"Ненулевых значений: {(spec > 0.01).sum()} / {spec.size}")

    # Текстовая мини-карта спектрограммы (8 строк × 20 столбцов)
    print(f"\nМини-карта (верх=высокие частоты, низ=низкие):")
    mini = spec[::max(1, spec.shape[0] // 8), ::max(1, spec.shape[1] // 20)]
    chars = " ░▒▓█"
    for row in mini[::-1]:
        line = "".join(chars[min(4, int(v * 5))] for v in row)
        print(f"  |{line}|")

    # ── 4. Токены ─────────────────────────────────────────────
    real_tokens = [t for t in tokens.tolist() if t not in (PAD_TOKEN,)]
    print(f"\n--- Токены ---")
    print(f"Всего слотов: {len(tokens)}  |  Реальных: {len(real_tokens)}")
    print(f"\nПоследовательность:")

    # Печатаем токены группами
    line = ""
    for i, tok in enumerate(real_tokens):
        s = decode_token(tok)
        if tok == EOS_TOKEN:
            line += f"  [{s}]"
            break
        elif tok == BOS_TOKEN:
            line += f"[{s}] "
        else:
            line += s + " "
        # Перенос строки каждые 8 токенов
        if (i + 1) % 8 == 0:
            print(f"  {line}")
            line = ""
    if line.strip():
        print(f"  {line}")

    # Статистика по типам токенов
    note_on = sum(1 for t in real_tokens if 3 <= t < 131)
    note_off = sum(1 for t in real_tokens if 131 <= t < 259)
    time_sh = sum(1 for t in real_tokens if 259 <= t < 359)
    velocity = sum(1 for t in real_tokens if 359 <= t < 391)
    print(f"\nСтатистика токенов:")
    print(f"  NOTE_ON:    {note_on}")
    print(f"  NOTE_OFF:   {note_off}")
    print(f"  TIME_SHIFT: {time_sh}")
    print(f"  VELOCITY:   {velocity}")

    # ── 5. Визуализация ───────────────────────────────────────
    if HAS_PLT and save_png:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        title = sample_dir.name
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            title += f"  |  {m.get('midi_file', '')}  seg={m.get('segment_idx', '')}  " \
                     f"[{m.get('start_sec', '')}–{m.get('end_sec', '')} сек]  " \
                     f"notes={m.get('n_notes', '')}  tokens={m.get('n_tokens', '')}"
        fig.suptitle(title, fontsize=9)

        im = ax1.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        ax1.set_title("Mel-спектрограмма (вход CNN-энкодера)")
        ax1.set_xlabel("Временной фрейм")
        ax1.set_ylabel("Mel-бин")
        plt.colorbar(im, ax=ax1)

        # Простой "piano roll" из токенов
        events = []
        t_ms = 0.0
        for tok in real_tokens:
            if tok == EOS_TOKEN:
                break
            if 259 <= tok < 359:
                t_ms += (tok - 259 + 1) * 10
            elif 3 <= tok < 131:
                events.append((t_ms / 1000.0, tok - 3))

        if events:
            times, pitches = zip(*events)
            ax2.scatter(times, pitches, s=20, c="steelblue", alpha=0.7)
            ax2.set_title("Ноты из токенов (правильный ответ модели)")
            ax2.set_xlabel("Время (сек)")
            ax2.set_ylabel("MIDI питч")
            ax2.set_xlim(0, 5.0)
        else:
            ax2.text(2.5, 60, "Нет нот в этом сегменте",
                     ha="center", va="center", color="gray")
            ax2.set_xlim(0, 5.0)

        plt.tight_layout()
        out_path = sample_dir / "verify.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\nКартинка сохранена: {out_path}")
    elif save_png and not HAS_PLT:
        print("\n[INFO] pip install matplotlib для сохранения картинок")

    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ручная проверка соответствия спектрограммы и токенов"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample_dir", help="Путь к конкретной папке sample_XXXXX")
    group.add_argument("--dataset_dir", help="Корень датасета (выбрать случайные)")
    parser.add_argument("--random", type=int, default=3,
                        help="Сколько случайных сэмплов проверить (при --dataset_dir)")
    parser.add_argument("--save_png", action="store_true",
                        help="Сохранить PNG-визуализацию в папку сэмпла")
    args = parser.parse_args()

    if args.sample_dir:
        check_one(Path(args.sample_dir), save_png=args.save_png)
    else:
        dataset_dir = Path(args.dataset_dir)
        samples = sorted([
            d for d in dataset_dir.iterdir()
            if d.is_dir() and (d / "spectrogram.npy").exists()
        ])
        if not samples:
            print(f"Сэмплы не найдены в {dataset_dir}")
            return
        chosen = random.sample(samples, min(args.random, len(samples)))
        print(f"Проверяем {len(chosen)} из {len(samples)} сэмплов:")
        for s in chosen:
            check_one(s, save_png=args.save_png)


if __name__ == "__main__":
    main()
