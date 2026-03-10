from __future__ import annotations
"""
MidiDataset
===========
Ожидает структуру данных на диске:

  dataset_root/
    ├── sample_001/
    │   ├── spectrogram.npy   — (N, F, T) float32, выход SpectrogramProcessor
    │   └── tokens.npy        — (seq_len,) int64,  выход MidiTokenizer
    ├── sample_002/
    │   └── ...
    └── ...

Скрипт prepare_dataset.py умеет автоматически генерировать эту структуру
из коллекции MIDI-файлов.
"""

"""
MidiSpectrogramDataset
=======================
Датасет читает пары (spectrogram.npy, tokens.npy) из dataset_root.

Нормировка спектрограммы:
  audio_processor.py уже сохраняет значения в [0, 1] (из dB шкалы).
  Если используется предобученный ResNet18 (imagenet_norm=True), датасет
  дополнительно применяет ImageNet-нормировку: mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225] по трём каналам (канал реплицируется × 3).

Форма выхода:
  spectrograms : (N, 3, F, T) если imagenet_norm=True  (для pretrained ResNet)
                 (N, 1, F, T) если imagenet_norm=False  (для custom CNN)
"""


from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

# ImageNet статистика (по каждому из 3 каналов)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class MidiSpectrogramDataset(Dataset):
    """
    Возвращает кортеж (spec, src_tokens, tgt_tokens, pad_mask).

    src_tokens = [BOS, t1, ..., tN]        — вход декодера (shifted right)
    tgt_tokens = [t1, ..., tN, EOS]        — цель (для CrossEntropyLoss)
    pad_mask   = True там где PAD          — передаётся в tgt_key_padding_mask
    """

    def __init__(
        self,
        dataset_root: str | Path,
        max_seq_len: int = 256,
        max_segments: int = 0,      # 0 = автоопределение из датасета
        max_freq_bins: int = 128,
        max_time_steps: int = 216,  # для Mel + hop=512: 5 сек ≈ 216 фреймов
        imagenet_norm: bool = True, # нужно если используется pretrained ResNet
    ):
        self.root          = Path(dataset_root)
        self.max_seq_len   = max_seq_len
        self.max_freq_bins = max_freq_bins
        self.max_time_steps = max_time_steps
        self.imagenet_norm = imagenet_norm

        # Собираем все образцы
        self.samples = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
            and (d / "spectrogram.npy").exists()
            and (d / "tokens.npy").exists()
        ])

        if not self.samples:
            raise RuntimeError(f"Нет данных в {self.root}. Запусти prepare_dataset.py")

        # Автоопределение max_segments
        _MAX_SEGMENTS_SAFE = 64   # жёсткий cap для 8 ГБ GPU (ResNet18 + d_model=256)
        if max_segments == 0:
            sizes = [np.load(d / "spectrogram.npy", mmap_mode="r").shape[0]
                     for d in self.samples]
            auto = int(np.max(sizes))
            if auto > _MAX_SEGMENTS_SAFE:
                print(f"  [WARN] max_segments автоопределён как {auto}, "
                      f"но ограничен до {_MAX_SEGMENTS_SAFE} во избежание OOM. "
                      f"Передайте max_segments явно чтобы изменить.")
                self.max_segments = _MAX_SEGMENTS_SAFE
            else:
                self.max_segments = auto
            print(f"  max_segments автоопределён: {self.max_segments} "
                  f"(min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)})")
        else:
            self.max_segments = max_segments

        print(f"Dataset: {len(self.samples)} образцов | "
              f"N={self.max_segments} сегм. × F={max_freq_bins} × T={max_time_steps}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_dir = self.samples[idx]

        # ── Спектрограмма ─────────────────────────────────────
        spec = np.load(sample_dir / "spectrogram.npy")   # (N, F, T) float32, значения [0,1]
        spec = self._pad_spec(spec)                       # → (max_segments, max_freq_bins, max_time_steps)

        if self.imagenet_norm:
            # Реплицируем 1 канал → 3 для претрейненного ResNet
            spec_t = torch.from_numpy(spec).unsqueeze(1).expand(-1, 3, -1, -1).clone()
            # (N, 3, F, T)
            spec_t = (spec_t - _IMAGENET_MEAN) / _IMAGENET_STD
        else:
            # (N, 1, F, T) — для кастомного CNN
            spec_t = torch.from_numpy(spec).unsqueeze(1)   # (N, 1, F, T)

        # ── Токены ────────────────────────────────────────────
        tokens = np.load(sample_dir / "tokens.npy").tolist()

        if not tokens or tokens[0] != BOS_TOKEN:
            tokens = [BOS_TOKEN] + tokens
        if tokens[-1] != EOS_TOKEN:
            tokens = tokens + [EOS_TOKEN]

        tokens = tokens[: self.max_seq_len + 1]
        src = tokens[:-1]
        tgt = tokens[1:]

        pad_len = self.max_seq_len - len(src)
        src = src + [PAD_TOKEN] * pad_len
        tgt = tgt + [PAD_TOKEN] * pad_len

        src_t  = torch.tensor(src, dtype=torch.long)
        tgt_t  = torch.tensor(tgt, dtype=torch.long)
        pad_mask = (src_t == PAD_TOKEN)

        return spec_t, src_t, tgt_t, pad_mask

    # ── Утилиты ───────────────────────────────────────────────
    def _pad_spec(self, spec: np.ndarray) -> np.ndarray:
        """Приводит (N, F, T) к (max_segments, max_freq_bins, max_time_steps)."""
        N, F, T = spec.shape
        Nt, Ft, Tt = self.max_segments, self.max_freq_bins, self.max_time_steps

        def pad_or_crop(arr, axis, size):
            s = arr.shape[axis]
            if s > size:
                idx = [slice(None)] * arr.ndim
                idx[axis] = slice(0, size)
                return arr[tuple(idx)]
            elif s < size:
                pad_shape = list(arr.shape)
                pad_shape[axis] = size - s
                return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=axis)
            return arr

        spec = pad_or_crop(spec, 0, Nt)
        spec = pad_or_crop(spec, 1, Ft)
        spec = pad_or_crop(spec, 2, Tt)
        return spec