"""
MIDI Event Tokenizer v2
=======================
Изменения по сравнению с v1:
  + SILENCE_TOKEN = 391  — явный токен тишины для пустых сегментов
  + VOCAB_SIZE = 392 (было 391)
  + encode_segment: если нет нот в окне -[BOS, SILENCE, EOS]
 
Словарь событий:
  PAD=0, BOS=1, EOS=2
  NOTE_ON_<0-127>   (128 токенов)   — нота нажата (pitch)
  NOTE_OFF_<0-127>  (128 токенов)   — нота отпущена (pitch)
  TIME_SHIFT_<1-100>(100 токенов)   — сдвиг времени (x * 10 мс)
  VELOCITY_<1-32>   (32 токена)     — динамика (velocity / 4)
  SILENCE           (1 токен)       — сегмент без нот (тишина)
 
Итого: 3 + 128 + 128 + 100 + 32 + 1 = 392 токена
 
ВАЖНО: SILENCE_TOKEN стоит ПОСЛЕ всех остальных, чтобы
       все существующие токены сохранили свои индексы.
       Единственное что меняется — VOCAB_SIZE (391 -392).
       Старые датасеты нужно пересоздать (prepare_dataset.py).
"""
 
from __future__ import annotations
from typing import List, Tuple
import numpy as np
 
try:
    import pretty_midi
except ImportError:
    pretty_midi = None
 
# ──────────────────────────────────────────────────────────────
#  Константы словаря
# ──────────────────────────────────────────────────────────────
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
 
_NOTE_ON_OFFSET = 3
_NOTE_OFF_OFFSET = _NOTE_ON_OFFSET + 128       # 131
_TIME_SHIFT_OFFSET = _NOTE_OFF_OFFSET + 128    # 259
_VELOCITY_OFFSET = _TIME_SHIFT_OFFSET + 100    # 359
 
# ── НОВОЕ: Токен тишины ──────────────────────────────────────
SILENCE_TOKEN = _VELOCITY_OFFSET + 32          # 391
 
VOCAB_SIZE = SILENCE_TOKEN + 1                  # 392
 
TIME_STEP_MS = 10       # 1 единица TIME_SHIFT = 10 мс
MAX_TIME_STEPS = 100    # максимальный шаг за один токен = 1 сек
VELOCITY_BINS = 32
 
 
# ──────────────────────────────────────────────────────────────
#  Вспомогательные функции кодирования
# ──────────────────────────────────────────────────────────────
def note_on_token(pitch: int) -> int:
    assert 0 <= pitch < 128
    return _NOTE_ON_OFFSET + pitch
 
 
def note_off_token(pitch: int) -> int:
    assert 0 <= pitch < 128
    return _NOTE_OFF_OFFSET + pitch
 
 
def time_shift_token(steps: int) -> int:
    steps = max(1, min(steps, MAX_TIME_STEPS))
    return _TIME_SHIFT_OFFSET + (steps - 1)
 
 
def velocity_token(velocity: int) -> int:
    bin_idx = max(1, min(int(velocity / 4) + 1, VELOCITY_BINS))
    return _VELOCITY_OFFSET + (bin_idx - 1)
 
 
# ──────────────────────────────────────────────────────────────
#  Декодирование токена -строка (для отладки)
# ──────────────────────────────────────────────────────────────
def decode_token(token: int) -> str:
    if token == PAD_TOKEN:      return "<PAD>"
    if token == BOS_TOKEN:      return "<BOS>"
    if token == EOS_TOKEN:      return "<EOS>"
    if token == SILENCE_TOKEN:  return "<SILENCE>"
    if _NOTE_ON_OFFSET <= token < _NOTE_OFF_OFFSET:
        return f"NOTE_ON({token - _NOTE_ON_OFFSET})"
    if _NOTE_OFF_OFFSET <= token < _TIME_SHIFT_OFFSET:
        return f"NOTE_OFF({token - _NOTE_OFF_OFFSET})"
    if _TIME_SHIFT_OFFSET <= token < _VELOCITY_OFFSET:
        return f"TIME_SHIFT({token - _TIME_SHIFT_OFFSET + 1})"
    if _VELOCITY_OFFSET <= token < SILENCE_TOKEN:
        return f"VELOCITY({(token - _VELOCITY_OFFSET + 1) * 4})"
    return f"<UNKNOWN:{token}>"
 
 
# ──────────────────────────────────────────────────────────────
#  Тип токена (для диагностики)
# ──────────────────────────────────────────────────────────────
def token_type(token: int) -> str:
    if token == PAD_TOKEN:      return "PAD"
    if token == BOS_TOKEN:      return "BOS"
    if token == EOS_TOKEN:      return "EOS"
    if token == SILENCE_TOKEN:  return "SILENCE"
    if _NOTE_ON_OFFSET  <= token < _NOTE_OFF_OFFSET:  return "NOTE_ON"
    if _NOTE_OFF_OFFSET <= token < _TIME_SHIFT_OFFSET: return "NOTE_OFF"
    if _TIME_SHIFT_OFFSET <= token < _VELOCITY_OFFSET: return "TIME_SHIFT"
    if _VELOCITY_OFFSET <= token < SILENCE_TOKEN:      return "VELOCITY"
    return "UNKNOWN"
 
 
# ──────────────────────────────────────────────────────────────
#  Внутренняя функция: собрать события из pretty_midi
# ──────────────────────────────────────────────────────────────
def _collect_events(midi_path: str) -> List[Tuple[float, str, int, int]]:
    """
    Возвращает список (time_sec, 'NOTE_ON'|'NOTE_OFF', pitch, velocity),
    отсортированный по времени. NOTE_OFF идут раньше NOTE_ON при одинаковом времени.
    """
    if pretty_midi is None:
        raise ImportError("pip install pretty_midi")
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            events.append((note.start, "NOTE_ON", note.pitch, note.velocity))
            events.append((note.end, "NOTE_OFF", note.pitch, 0))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "NOTE_OFF" else 1))
    return events
 
 
# ──────────────────────────────────────────────────────────────
#  Внутренняя функция: превратить список событий в токены
# ──────────────────────────────────────────────────────────────
def _events_to_tokens(
        events: List[Tuple[float, str, int, int]],
        max_tokens: int,
) -> List[int]:
    """
    Конвертирует список событий в токен-последовательность.
    Время событий должно быть уже смещено (т.е. начинаться с 0).
    """
    tokens: List[int] = [BOS_TOKEN]
    current_time_ms = 0.0
 
    for (time_sec, etype, pitch, velocity) in events:
        event_ms = time_sec * 1000.0
        delta_ms = event_ms - current_time_ms
        steps_total = max(0, round(delta_ms / TIME_STEP_MS))
 
        # Вставляем TIME_SHIFT токены (разбиваем на чанки по MAX_TIME_STEPS)
        while steps_total > 0:
            chunk = min(steps_total, MAX_TIME_STEPS)
            tokens.append(time_shift_token(chunk))
            steps_total -= chunk
 
        current_time_ms = event_ms
 
        if etype == "NOTE_ON":
            tokens.append(velocity_token(velocity))
            tokens.append(note_on_token(pitch))
        else:
            tokens.append(note_off_token(pitch))
 
        if len(tokens) >= max_tokens - 1:
            break
 
    tokens.append(EOS_TOKEN)
    return tokens
 
 
# ──────────────────────────────────────────────────────────────
#  Главный класс токенизатора
# ──────────────────────────────────────────────────────────────
class MidiTokenizer:
 
    def __init__(self, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
 
    # ── Кодирование всего трека ───────────────────────────────
    def encode(self, midi_path: str) -> List[int]:
        """Кодирует весь MIDI-файл в одну последовательность токенов."""
        events = _collect_events(midi_path)
        if not events:
            return [BOS_TOKEN, SILENCE_TOKEN, EOS_TOKEN]
        return _events_to_tokens(events, self.max_seq_len)
 
    # ── КЛЮЧЕВОЙ МЕТОД: кодирование одного временно́го сегмента ──
    def encode_segment(
            self,
            midi_path: str,
            start_sec: float,
            end_sec: float,
    ) -> List[int]:
        """
        Кодирует ТОЛЬКО события в окне [start_sec, end_sec).
 
        ИЗМЕНЕНИЕ v2: если нот в окне нет — возвращает [BOS, SILENCE, EOS]
        вместо [BOS, EOS]. Это позволяет модели явно предсказать
        «тишину», а не путать пустую последовательность с паддингом.
        """
        all_events = _collect_events(midi_path)
 
        # Множество питчей нот, которые НАЧАЛИСЬ в этом окне
        notes_started_in_window = set()
 
        # Отбираем события в окне
        filtered: List[Tuple[float, str, int, int]] = []
        for (t, etype, pitch, velocity) in all_events:
            if etype == "NOTE_ON" and start_sec <= t < end_sec:
                notes_started_in_window.add(pitch)
                # Сдвигаем время: 0 = начало сегмента
                filtered.append((t - start_sec, etype, pitch, velocity))
            elif etype == "NOTE_OFF" and pitch in notes_started_in_window:
                t_clipped = min(t, end_sec) - start_sec
                filtered.append((t_clipped, etype, pitch, 0))
                notes_started_in_window.discard(pitch)
 
        # ── НОВОЕ: Если нет нот -явный SILENCE ──────────────
        if not filtered:
            return [BOS_TOKEN, SILENCE_TOKEN, EOS_TOKEN]
 
        # Сортируем ещё раз после смещения
        filtered.sort(key=lambda e: (e[0], 0 if e[1] == "NOTE_OFF" else 1))
 
        return _events_to_tokens(filtered, self.max_seq_len)
 
    # ── Декодирование ─────────────────────────────────────────
    def decode(self, tokens: List[int]) -> List[Tuple[float, str, int, int]]:
        """Токены -список (time_sec, event_type, pitch, velocity)."""
        events = []
        current_time_ms = 0.0
        current_velocity = 64
 
        for token in tokens:
            if token == EOS_TOKEN:
                break
            if token in (PAD_TOKEN, BOS_TOKEN, SILENCE_TOKEN):
                continue
            if _TIME_SHIFT_OFFSET <= token < _VELOCITY_OFFSET:
                current_time_ms += (token - _TIME_SHIFT_OFFSET + 1) * TIME_STEP_MS
            elif _VELOCITY_OFFSET <= token < SILENCE_TOKEN:
                current_velocity = (token - _VELOCITY_OFFSET + 1) * 4
            elif _NOTE_ON_OFFSET <= token < _NOTE_OFF_OFFSET:
                events.append((current_time_ms / 1000.0, "NOTE_ON",
                               token - _NOTE_ON_OFFSET, current_velocity))
            elif _NOTE_OFF_OFFSET <= token < _TIME_SHIFT_OFFSET:
                events.append((current_time_ms / 1000.0, "NOTE_OFF",
                               token - _NOTE_OFF_OFFSET, 0))
        return events
 
    # ── Паддинг ───────────────────────────────────────────────
    def pad(self, tokens: List[int], length: int) -> List[int]:
        return (tokens[:length] + [PAD_TOKEN] * length)[:length]
 
    def to_numpy(self, tokens: List[int], length: int) -> np.ndarray:
        return np.array(self.pad(tokens, length), dtype=np.int64)
    