"""
prepare_dataset.py
==================
Генерирует датасет из папки с MIDI-файлами.

Запуск:
    python prepare_dataset.py \
        --midi_dir   /path/to/midi_files \
        --output_dir /path/to/dataset \
        --soundfont  /path/to/soundfont.sf2

Структура выходных данных:
    output_dir/
        sample_0000/
            spectrogram.npy
            tokens.npy
        sample_0001/
            ...
"""

import argparse
import sys
from pathlib import Path

# Корень проекта (родитель Encoder_Decoder/) — чтобы midi_to_fft был виден как пакет
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
# Сама папка midi_to_fft — чтобы работали голые импорты внутри пакета
# (from config import ..., from midi_renderer import ... и т.д.)
sys.path.insert(0, str(_PROJECT_ROOT / "midi_to_fft"))

import numpy as np
from tqdm import tqdm

from midi_to_fft import AudioConfig, MidiToFFTMap
from tokenizer import MidiTokenizer


def prepare(
    midi_dir: str,
    output_dir: str,
    soundfont_path: str,
    max_seq_len: int = 512,
):
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    if not midi_files:
        print(f"MIDI-файлы не найдены в {midi_dir}")
        return

    print(f"Найдено {len(midi_files)} MIDI-файлов")

    conf = AudioConfig()
    pipeline = MidiToFFTMap(soundfont_path=soundfont_path, config=conf)
    tokenizer = MidiTokenizer(max_seq_len=max_seq_len)

    ok, err = 0, 0
    for i, midi_path in enumerate(tqdm(midi_files, desc="Обработка")):
        try:
            # 1. Спектрограмма
            spectrogram = pipeline.process(str(midi_path))  # (N, F, T)

            # 2. Токены
            tokens = tokenizer.encode(str(midi_path))       # list[int]
            tokens_arr = tokenizer.to_numpy(tokens, max_seq_len)

            # 3. Сохранение
            sample_dir = output_dir / f"sample_{i:04d}"
            sample_dir.mkdir(exist_ok=True)
            np.save(sample_dir / "spectrogram.npy", spectrogram)
            np.save(sample_dir / "tokens.npy", tokens_arr)
            ok += 1

        except Exception as e:
            print(f"\n[WARN] {midi_path.name}: {e}")
            err += 1

    print(f"\nГотово: {ok} успешно, {err} ошибок")
    print(f"Датасет сохранён в {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--soundfont",  required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    prepare(
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        soundfont_path=args.soundfont,
        max_seq_len=args.max_seq_len,
    )
