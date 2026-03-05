"""
inference.py
============
Генерация нотной партитуры из аудиофайла.

Пример запуска:
    python inference.py \
        --audio     /path/to/audio.wav \
        --model_pt  checkpoints/best_model.pt \
        --soundfont soundfonts/FluidR3_GM.sf2 \
        --output    generated.mid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "midi_to_fft"))

from midi_to_fft import AudioConfig, SpectrogramProcessor

from model import ScoreGenerationModel
from tokenizer import MidiTokenizer, VOCAB_SIZE, decode_token

# Для записи MIDI: pretty_midi
try:
    import pretty_midi
except ImportError:
    pretty_midi = None


def load_audio_spectrogram(audio_path: str, config: AudioConfig) -> np.ndarray:
    """Загружает WAV/MP3 и строит спектрограмму (N, F, T)."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    processor = SpectrogramProcessor(config)
    return processor.compute(audio)


def tokens_to_midi(events, output_path: str, instrument_program: int = 0):
    """Конвертирует список событий токенизатора в MIDI-файл."""
    if pretty_midi is None:
        print("[WARN] pretty_midi не установлен. pip install pretty_midi")
        return

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=instrument_program)

    # Активные ноты: pitch → (start_time, velocity)
    active: dict[int, tuple[float, int]] = {}
    current_velocity = 64

    for (time_sec, etype, pitch, velocity) in events:
        if etype == "NOTE_ON":
            active[pitch] = (time_sec, velocity)
        elif etype == "NOTE_OFF":
            if pitch in active:
                start, vel = active.pop(pitch)
                end = max(start + 0.05, time_sec)   # минимальная длительность
                note = pretty_midi.Note(
                    velocity=vel, pitch=pitch,
                    start=start, end=end
                )
                inst.notes.append(note)

    # Закрываем незакрытые ноты
    max_time = max((n.end for n in inst.notes), default=0.0)
    for pitch, (start, vel) in active.items():
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch,
            start=start, end=max_time + 0.1
        ))

    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"MIDI сохранён: {output_path}  ({len(inst.notes)} нот)")


def run(
    audio_path: str,
    model_pt: str,
    output_path: str,
    soundfont: str | None = None,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    d_model: int = 128,
    nhead: int = 8,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # ── 1. Спектрограмма ─────────────────────────────────────
    conf = AudioConfig()
    spec = load_audio_spectrogram(audio_path, conf)   # (N, F, T)
    print(f"Спектрограмма: {spec.shape}")

    # Нормировка
    mn, mx = spec.min(), spec.max()
    if mx - mn > 1e-6:
        spec = 2.0 * (spec - mn) / (mx - mn) - 1.0

    spec_tensor = torch.from_numpy(spec).unsqueeze(0).to(device)  # (1, N, F, T)

    # ── 2. Модель ────────────────────────────────────────────
    model = ScoreGenerationModel(
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        vocab_size=VOCAB_SIZE,
        max_seq_len=max_len,
    ).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()
    print("Модель загружена")

    # ── 3. Генерация ─────────────────────────────────────────
    with torch.no_grad():
        generated = model.generate(
            spectrograms=spec_tensor,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
        )

    token_list = generated[0].cpu().tolist()
    print(f"Сгенерировано {len(token_list)} токенов")
    print("Первые 20:", [decode_token(t) for t in token_list[:20]])

    # ── 4. Декодирование в MIDI ──────────────────────────────
    tokenizer = MidiTokenizer(max_seq_len=max_len)
    events    = tokenizer.decode(token_list)
    tokens_to_midi(events, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",       required=True)
    parser.add_argument("--model_pt",    required=True)
    parser.add_argument("--output",      default="generated.mid")
    parser.add_argument("--soundfont",   default=None)
    parser.add_argument("--max_len",     type=int,   default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k",       type=int,   default=50)
    args = parser.parse_args()

    run(
        audio_path=args.audio,
        model_pt=args.model_pt,
        output_path=args.output,
        soundfont=args.soundfont,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )
