"""
prepare_dataset.py
==================
Генерирует датасет из папки с MIDI-файлами.

Структура выходных данных:
    output_dir/
        sample_000000/          <-- сегмент 0 трека midi_001.mid  (0–5 сек)
            spectrogram.npy     — (n_mels, time_steps) float32
            tokens.npy          — (max_seq_len,) int64
            meta.json           — {midi_file, segment_idx, start_sec, end_sec, n_notes}
        sample_000001/          <-- сегмент 1 трека midi_001.mid  (5–10 сек)
            ...
        sample_000012/          <-- сегмент 0 трека midi_002.mid  (0–5 сек)
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

Ускорение за счёт multiprocessing: каждый воркер обрабатывает
один MIDI-файл независимо (рендеринг + спектрограмма + токенизация).


Запуск:
    python prepare_dataset.py \\
        --midi_dir   ./midi_trimmed \\
        --output_dir ./dataset \\
        --soundfont  FluidR3_GM.sf2

    # Задать число воркеров явно (по умолчанию = число ядер - 1):
    python prepare_dataset.py ... --workers 8

    # Только сегменты с хотя бы N нотами:
    python prepare_dataset.py ... --min_notes 2

    # Проверить без сохранения:
    python prepare_dataset.py ... --verify_only
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "midi_to_fft"))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# ──────────────────────────────────────────────────────────────
#  Функция-воркер (выполняется в отдельном процессе)
# ──────────────────────────────────────────────────────────────
def _worker(args: tuple) -> dict:
    """
    Обрабатывает один MIDI-файл.
    Имена папок: sample_{file_idx:05d}_{seg_idx:03d}/
    Это гарантирует уникальность без общего счётчика между процессами.
    """
    (midi_path_str, output_dir_str, soundfont_path,
     max_seq_len, min_notes, verify_only, file_idx) = args

    # Импорты внутри воркера — каждый процесс инициализирует своё окружение
    from midi_to_fft import AudioConfig, MidiToFFTMap
    from tokenizer import MidiTokenizer, PAD_TOKEN
    import pretty_midi
    import numpy as np

    midi_path = Path(midi_path_str)
    output_dir = Path(output_dir_str)
    config = AudioConfig()
    tokenizer = MidiTokenizer(max_seq_len=max_seq_len)
    seg_sec = config.segment_size_sec

    result = {
        "file": midi_path.name,
        "saved": 0,
        "skipped": 0,
        "error": None,
        "segments": [],
    }

    try:
        # 1. Рендеринг + спектрограмма всего трека
        pipeline = MidiToFFTMap(soundfont_path=soundfont_path, config=config)
        spectrograms = pipeline.process(str(midi_path))  # (N, F, T)
        N = spectrograms.shape[0]

        # 2. Читаем ноты один раз для всего трека
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        all_notes = []
        for inst in pm.instruments:
            all_notes.extend(inst.notes)
        all_notes.sort(key=lambda n: n.start)

        # 3. Обрабатываем каждый сегмент
        for seg_idx in range(N):
            t_start = seg_idx * seg_sec
            t_end = t_start + seg_sec

            n_notes = sum(1 for n in all_notes if t_start <= n.start < t_end)

            if n_notes < min_notes:
                result["skipped"] += 1
                continue

            tokens = tokenizer.encode_segment(str(midi_path), t_start, t_end)
            tokens_arr = tokenizer.to_numpy(tokens, max_seq_len)
            n_real_tok = int((tokens_arr != PAD_TOKEN).sum())

            if verify_only:
                result["segments"].append({
                    "seg_idx": seg_idx,
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_notes": n_notes,
                    "n_tokens": n_real_tok,
                })
                continue

            # Уникальное имя без общего счётчика
            sample_dir = output_dir / f"sample_{file_idx:05d}_{seg_idx:03d}"
            sample_dir.mkdir(exist_ok=True)

            np.save(sample_dir / "spectrogram.npy", spectrograms[seg_idx])
            np.save(sample_dir / "tokens.npy", tokens_arr)
            (sample_dir / "meta.json").write_text(json.dumps({
                "midi_file": midi_path.name,
                "segment_idx": seg_idx,
                "start_sec": round(t_start, 3),
                "end_sec": round(t_end, 3),
                "n_notes": n_notes,
                "n_tokens": n_real_tok,
            }, ensure_ascii=False, indent=2))

            result["saved"] += 1

    except Exception as e:
        result["error"] = str(e)

    return result


# ──────────────────────────────────────────────────────────────
#  Главная функция
# ──────────────────────────────────────────────────────────────
def prepare(
        midi_dir: str,
        output_dir: str,
        soundfont_path: str,
        max_seq_len: int = 256,
        min_notes: int = 0,
        workers: int = 0,
        verify_only: bool = False,
):
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)

    if not verify_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(
        list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    )
    if not midi_files:
        print(f"MIDI-файлы не найдены в {midi_dir}")
        return

    n_workers = workers if workers > 0 else max(1, mp.cpu_count() - 1)
    print(f"Найдено:  {len(midi_files)} MIDI-файлов")
    print(f"Ядра:  {n_workers}  (CPU ядер: {mp.cpu_count()})")
    print(f"Сегмент:  5 сек  |  min_notes={min_notes}  |  max_seq_len={max_seq_len}\n")

    tasks = [
        (str(p), str(output_dir), soundfont_path,
         max_seq_len, min_notes, verify_only, i)
        for i, p in enumerate(midi_files)
    ]

    t0 = time.time()
    total_saved = 0
    total_skipped = 0
    total_errors = 0

    with mp.Pool(processes=n_workers) as pool:
        for result in tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc="Обработка",
                unit="midi",
        ):
            if result["error"]:
                total_errors += 1
                print(f"\n[ОШИБКА] {result['file']}: {result['error']}")
            else:
                total_saved += result["saved"]
                total_skipped += result["skipped"]

            if verify_only and result["segments"]:
                print(f"\n{result['file']}:")
                for s in result["segments"]:
                    print(f"  seg {s['seg_idx']:2d}"
                          f" [{s['t_start']:.1f}–{s['t_end']:.1f}s]"
                          f"  notes={s['n_notes']:3d}  tokens={s['n_tokens']:3d}")

    elapsed = time.time() - t0
    rate = len(midi_files) / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 55}")
    if verify_only:
        print("Проверка завершена (файлы не записывались)")
    else:
        print(f"Готово за {elapsed:.1f} сек  ({rate:.1f} файлов/сек)")
        print(f"  Сегментов сохранено:  {total_saved}")
        print(f"  Пропущено (нет нот):  {total_skipped}")
        print(f"  Ошибок:               {total_errors}")
        print(f"  Датасет:              {output_dir}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()  # нужно для Windows

    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--soundfont", required=True)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--min_notes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Число процессов (0 = авто: ядра CPU - 1)")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    prepare(
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        soundfont_path=args.soundfont,
        max_seq_len=args.max_seq_len,
        min_notes=args.min_notes,
        workers=args.workers,
        verify_only=args.verify_only,
    )
