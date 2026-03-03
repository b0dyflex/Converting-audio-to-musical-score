"""
MIDI Event Tokenizer
====================
Словарь событий:
  PAD, BOS, EOS                         — служебные токены
  NOTE_ON_<0-127>   (128 токенов)       — нота нажата (pitch)
  NOTE_OFF_<0-127>  (128 токенов)       — нота отпущена (pitch)
  TIME_SHIFT_<1-100>(100 токенов)       — сдвиг времени (x * TIME_STEP_MS мс)
  VELOCITY_<1-32>   (32 токена)         — динамика (квантование velocity / 4)

Итого словарь: 3 + 128 + 128 + 100 + 32 = 391 токен
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None  # опционально для чтения MIDI

# ──────────────────────────────────────────────────────────────
#  Константы
# ──────────────────────────────────────────────────────────────
PAD_TOKEN  = 0
BOS_TOKEN  = 1
EOS_TOKEN  = 2

_NOTE_ON_OFFSET   = 3
_NOTE_OFF_OFFSET  = _NOTE_ON_OFFSET + 128      
_TIME_SHIFT_OFFSET = _NOTE_OFF_OFFSET + 128   
_VELOCITY_OFFSET  = _TIME_SHIFT_OFFSET + 100   

VOCAB_SIZE = _VELOCITY_OFFSET + 32           

TIME_STEP_MS = 10
MAX_TIME_STEPS = 100
VELOCITY_BINS = 32


# ──────────────────────────────────────────────────────────────
#  Вспомогательные функции кодирования токенов
# ──────────────────────────────────────────────────────────────
def note_on_token(pitch: int) -> int:
    assert 0 <= pitch < 128
    return _NOTE_ON_OFFSET + pitch

def note_off_token(pitch: int) -> int:
    assert 0 <= pitch < 128
    return _NOTE_OFF_OFFSET + pitch

def time_shift_token(steps: int) -> int:
    """steps: 1..100"""
    steps = max(1, min(steps, MAX_TIME_STEPS))
    return _TIME_SHIFT_OFFSET + (steps - 1)

def velocity_token(velocity: int) -> int:
    """velocity: 0..127  →  bin: 1..32"""
    bin_idx = max(1, min(int(velocity / 4) + 1, VELOCITY_BINS))
    return _VELOCITY_OFFSET + (bin_idx - 1)


# ──────────────────────────────────────────────────────────────
#  Декодирование токена - событие (для отладки)
# ──────────────────────────────────────────────────────────────
def decode_token(token: int) -> str:
    if token == PAD_TOKEN:   return "<PAD>"
    if token == BOS_TOKEN:   return "<BOS>"
    if token == EOS_TOKEN:   return "<EOS>"
    if _NOTE_ON_OFFSET <= token < _NOTE_OFF_OFFSET:
        return f"NOTE_ON({token - _NOTE_ON_OFFSET})"
    if _NOTE_OFF_OFFSET <= token < _TIME_SHIFT_OFFSET:
        return f"NOTE_OFF({token - _NOTE_OFF_OFFSET})"
    if _TIME_SHIFT_OFFSET <= token < _VELOCITY_OFFSET:
        return f"TIME_SHIFT({token - _TIME_SHIFT_OFFSET + 1})"
    if _VELOCITY_OFFSET <= token < VOCAB_SIZE:
        return f"VELOCITY({(token - _VELOCITY_OFFSET + 1) * 4})"
    return f"<UNKNOWN:{token}>"


# ──────────────────────────────────────────────────────────────
#  Главный класс токенизатора
# ──────────────────────────────────────────────────────────────
class MidiTokenizer:
    """
    Преобразует MIDI-файл в последовательность токенов и обратно.

    Алгоритм:
      - Читаем все ноты и сортируем по времени начала
      - Для каждого события вычисляем delta-time в единицах TIME_STEP_MS
      - Если delta > 0 — вставляем один или несколько TIME_SHIFT токенов
      - Затем VELOCITY + NOTE_ON при нажатии, NOTE_OFF при отпускании
    """

    def __init__(self, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len

    # ── Кодирование ──────────────────────────────────────────
    def encode(self, midi_path: str) -> List[int]:
        if pretty_midi is None:
            raise ImportError("pip install pretty_midi")

        pm = pretty_midi.PrettyMIDI(midi_path)

        # Собираем все события (time_sec, type, pitch, velocity)
        events: List[Tuple[float, str, int, int]] = []
        for instrument in pm.instruments:
            for note in instrument.notes:
                events.append((note.start, "NOTE_ON",  note.pitch, note.velocity))
                events.append((note.end,   "NOTE_OFF", note.pitch, 0))

        events.sort(key=lambda e: (e[0], 0 if e[1] == "NOTE_OFF" else 1))

        tokens = [BOS_TOKEN]
        current_time_ms = 0.0

        for (time_sec, etype, pitch, velocity) in events:
            event_time_ms = time_sec * 1000.0
            delta_ms = event_time_ms - current_time_ms

            # Вставляем TIME_SHIFT токены
            steps_total = max(0, round(delta_ms / TIME_STEP_MS))
            while steps_total > 0:
                chunk = min(steps_total, MAX_TIME_STEPS)
                tokens.append(time_shift_token(chunk))
                steps_total -= chunk

            current_time_ms = event_time_ms

            if etype == "NOTE_ON":
                tokens.append(velocity_token(velocity))
                tokens.append(note_on_token(pitch))
            else:
                tokens.append(note_off_token(pitch))

            if len(tokens) >= self.max_seq_len - 1:
                break

        tokens.append(EOS_TOKEN)
        return tokens

    # ── Декодирование (токены → список событий) ──────────────
    def decode(self, tokens: List[int]) -> List[Tuple[float, str, int, int]]:
        """
        Возвращает список (time_sec, event_type, pitch, velocity).
        event_type: 'NOTE_ON' | 'NOTE_OFF'
        """
        events = []
        current_time_ms = 0.0
        current_velocity = 64

        for token in tokens:
            if token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                if token == EOS_TOKEN:
                    break
                continue

            if _TIME_SHIFT_OFFSET <= token < _VELOCITY_OFFSET:
                steps = token - _TIME_SHIFT_OFFSET + 1
                current_time_ms += steps * TIME_STEP_MS

            elif _VELOCITY_OFFSET <= token < VOCAB_SIZE:
                current_velocity = (token - _VELOCITY_OFFSET + 1) * 4

            elif _NOTE_ON_OFFSET <= token < _NOTE_OFF_OFFSET:
                pitch = token - _NOTE_ON_OFFSET
                events.append((current_time_ms / 1000.0, "NOTE_ON", pitch, current_velocity))

            elif _NOTE_OFF_OFFSET <= token < _TIME_SHIFT_OFFSET:
                pitch = token - _NOTE_OFF_OFFSET
                events.append((current_time_ms / 1000.0, "NOTE_OFF", pitch, 0))

        return events

    # ── Паддинг / обрезка до фиксированной длины ─────────────
    def pad(self, tokens: List[int], length: int) -> List[int]:
        tokens = tokens[:length]
        return tokens + [PAD_TOKEN] * (length - len(tokens))

    def to_numpy(self, tokens: List[int], length: int) -> np.ndarray:
        return np.array(self.pad(tokens, length), dtype=np.int64)
