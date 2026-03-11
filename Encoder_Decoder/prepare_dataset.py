"""
prepare_dataset.py
==================
Генерирует датасет из папки с MIDI-файлами.

Структура выходных данных:
    output_dir/
        sample_000000/          ← сегмент 0 трека midi_001.mid  (0–5 сек)
            spectrogram.npy     — (n_mels, time_steps) float32
            tokens.npy          — (max_seq_len,) int64
            meta.json           — {midi_file, segment_idx, start_sec, end_sec, n_notes}
        sample_000001/          ← сегмент 1 трека midi_001.mid  (5–10 сек)
            ...
        sample_000012/          ← сегмент 0 трека midi_002.mid  (0–5 сек)
            ...

Запуск:
    python prepare_dataset.py \\
        --midi_dir   ./midi \\
        --output_dir ./dataset \\
        --soundfont  FluidR3_GM.sf2

    # Только сегменты с хотя бы 2 нотами:
    python prepare_dataset.py ... --min_notes 2

    # Проверить готовый датасет:
    python prepare_dataset.py ... --verify_only
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "midi_to_fft"))

import numpy as np
from tqdm import tqdm

from midi_to_fft import AudioConfig, MidiToFFTMap
from tokenizer import MidiTokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


# ──────────────────────────────────────────────────────────────
#  Утилита: сколько нот начинается в окне [t_start, t_end)
# ──────────────────────────────────────────────────────────────
def count_notes_in_window(midi_path: str, t_start: float, t_end: float) -> int:
    if pretty_midi is None:
        return -1
    pm = pretty_midi.PrettyMIDI(midi_path)
    count = 0
    for inst in pm.instruments:
        for note in inst.notes:
            if t_start <= note.start < t_end:
                count += 1
    return count


# ──────────────────────────────────────────────────────────────
#  Основная функция
# ──────────────────────────────────────────────────────────────
def prepare(
        midi_dir: str,
        output_dir: str,
        soundfont_path: str,
        max_seq_len: int = 256,
        min_notes: int = 1,  # пропускать сегменты с меньшим числом нот
        verify_only: bool = False,
):
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    if not midi_files:
        print(f"MIDI-файлы не найдены в {midi_dir}")
        return

    print(f"Найдено {len(midi_files)} MIDI-файлов")

    config = AudioConfig()
    pipeline = MidiToFFTMap(soundfont_path=soundfont_path, config=config)
    tokenizer = MidiTokenizer(max_seq_len=max_seq_len)

    segment_sec = config.segment_size_sec  # 5.0 по умолчанию

    total_samples = 0
    total_skipped = 0
    total_errors = 0
    sample_idx = 0

    for midi_path in tqdm(midi_files, desc="MIDI-файлы"):
        try:
            # ── 1. Спектрограмма всего трека ──────────────────
            # pipeline.process возвращает (N, F, T) — N сегментов
            spectrograms = pipeline.process(str(midi_path))  # (N, n_mels, time_steps)
            N = spectrograms.shape[0]

            # ── 2. Для каждого сегмента — свои токены ─────────
            for seg_idx in range(N):
                t_start = seg_idx * segment_sec
                t_end = t_start + segment_sec

                # Считаем ноты в окне (для фильтрации и meta)
                n_notes = count_notes_in_window(str(midi_path), t_start, t_end)

                # Пропускаем почти пустые сегменты
                if n_notes < min_notes:
                    total_skipped += 1
                    continue

                if verify_only:
                    # Только выводим информацию, не сохраняем
                    tokens = tokenizer.encode_segment(str(midi_path), t_start, t_end)
                    n_real_tokens = sum(1 for t in tokens
                                        if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN))
                    print(f"  {midi_path.name} | seg {seg_idx:2d} "
                          f"[{t_start:.1f}–{t_end:.1f}s] | "
                          f"notes={n_notes:3d}  tokens={n_real_tokens:3d}")
                    continue

                # ── Кодируем токены только для этого сегмента ──
                tokens = tokenizer.encode_segment(str(midi_path), t_start, t_end)
                tokens_arr = tokenizer.to_numpy(tokens, max_seq_len)

                # ── Спектрограмма этого сегмента (F, T) ─────────
                spec_2d = spectrograms[seg_idx]  # (n_mels, time_steps)

                # ── Сохраняем ──────────────────────────────────
                sample_dir = output_dir / f"sample_{sample_idx:06d}"
                sample_dir.mkdir(exist_ok=True)

                np.save(sample_dir / "spectrogram.npy", spec_2d)
                np.save(sample_dir / "tokens.npy", tokens_arr)

                meta = {
                    "midi_file": midi_path.name,
                    "segment_idx": seg_idx,
                    "start_sec": round(t_start, 3),
                    "end_sec": round(t_end, 3),
                    "n_notes": n_notes,
                    "n_tokens": int((tokens_arr != PAD_TOKEN).sum()),
                }
                (sample_dir / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2)
                )

                sample_idx += 1
                total_samples += 1

        except Exception as e:
            print(f"\n[ОШИБКА] {midi_path.name}: {e}")
            total_errors += 1

    if verify_only:
        print(f"\nПроверка завершена (файлы не сохранялись)")
    else:
        print(f"\n{'=' * 55}")
        print(f"Готово: {total_samples} сегментов сохранено")
        print(f"  Пропущено (нет нот): {total_skipped}")
        print(f"  Ошибок:              {total_errors}")
        print(f"  Датасет:             {output_dir}")
        print(f"  Сегмент:             {segment_sec} сек")
        print(f"  Формат:              spectrogram (F,T) + tokens (max_seq={max_seq_len})")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Создаёт датасет: один sample = один 5-секундный сегмент"
    )
    parser.add_argument("--midi_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--soundfont", required=True)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--min_notes", type=int, default=1,
                        help="Пропускать сегменты с меньшим числом нот")
    parser.add_argument("--verify_only", action="store_true",
                        help="Только показать что будет в датасете, не сохранять")
    args = parser.parse_args()

    prepare(
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        soundfont_path=args.soundfont,
        max_seq_len=args.max_seq_len,
        min_notes=args.min_notes,
        verify_only=args.verify_only,
    )
