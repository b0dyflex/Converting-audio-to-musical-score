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

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


class MidiSpectrogramDataset(Dataset):
    """
    Каждый элемент:
        spectrograms : FloatTensor  (N, F, T)
        src_tokens   : LongTensor  (tgt_len,)  — токены со смещением вправо (для входа декодера)
        tgt_tokens   : LongTensor  (tgt_len,)  — целевые токены (для вычисления loss)
        padding_mask : BoolTensor  (tgt_len,)  — True там где PAD

    src_tokens = [BOS, t1, t2, ..., tN]
    tgt_tokens = [t1,  t2, ..., tN, EOS]
    """

    def __init__(
        self,
        dataset_root: str | Path,
        max_seq_len: int = 512,
        max_segments: int = 64,    # ← максимальное число сегментов N
        max_freq_bins: int = 128,
        max_time_steps: int = 399,
        normalize_spectrogram: bool = True,
    ):
        self.root = Path(dataset_root)
        self.max_seq_len = max_seq_len
        self.max_segments = max_segments    # ← сохраняем
        self.max_freq_bins = max_freq_bins
        self.max_time_steps = max_time_steps
        self.normalize = normalize_spectrogram

        # Собираем все папки с парными файлами
        self.samples = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
            and (d / "spectrogram.npy").exists()
            and (d / "tokens.npy").exists()
        ])

        if len(self.samples) == 0:
            raise RuntimeError(f"Нет данных в {self.root}. Запусти prepare_dataset.py")

        # Если max_segments не задан явно — вычисляем из датасета
        if max_segments == 0:
            sizes = [np.load(d / "spectrogram.npy", mmap_mode="r").shape[0]
                     for d in self.samples]
            self.max_segments = int(np.max(sizes))
            print(f"  max_segments автоопределён: {self.max_segments} "
                  f"(min={min(sizes)}, max={max(sizes)})")
        else:
            self.max_segments = max_segments

        print(f"Dataset: {len(self.samples)} образцов в {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_dir = self.samples[idx]

        # ── Спектрограмма ────────────────────────────────────
        spec = np.load(sample_dir / "spectrogram.npy")   # (N, F, T)
        spec = self._pad_spectrogram(spec)
        if self.normalize:
            spec = self._normalize(spec)
        spec_tensor = torch.from_numpy(spec)              # FloatTensor (N, F, T)

        # ── Токены ───────────────────────────────────────────
        tokens = np.load(sample_dir / "tokens.npy").tolist()  # list[int]

        # Убеждаемся, что есть BOS и EOS
        if tokens[0] != BOS_TOKEN:
            tokens = [BOS_TOKEN] + tokens
        if tokens[-1] != EOS_TOKEN:
            tokens = tokens + [EOS_TOKEN]

        # Обрезаем до max_seq_len + 1 (нам нужен сдвиг на 1)
        tokens = tokens[:self.max_seq_len + 1]

        src = tokens[:-1]   # [BOS, t1, ..., tN]
        tgt = tokens[1:]    # [t1,  t2, ..., EOS]

        # Паддинг до max_seq_len
        seq_len = len(src)
        src = src + [PAD_TOKEN] * (self.max_seq_len - seq_len)
        tgt = tgt + [PAD_TOKEN] * (self.max_seq_len - seq_len)

        src_tensor = torch.tensor(src, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)
        pad_mask   = (src_tensor == PAD_TOKEN)   # BoolTensor

        return spec_tensor, src_tensor, tgt_tensor, pad_mask

    def _pad_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        Приводит спектрограмму к фиксированной форме (max_segments, max_freq_bins, max_time_steps).
        Лишнее обрезается, недостающее дополняется нулями.
        """
        N, F, T = spec.shape
        N_target = self.max_segments
        F_target = self.max_freq_bins
        T_target = self.max_time_steps

        # ── 1. Обрезка / паддинг по N (число сегментов) ─────
        if N > N_target:
            spec = spec[:N_target, :, :]
        elif N < N_target:
            pad = np.zeros((N_target - N, F, T), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=0)

        N = spec.shape[0]   # теперь N == N_target

        # ── 2. Обрезка / паддинг по F ───────────────────────
        if F > F_target:
            spec = spec[:, :F_target, :]
        elif F < F_target:
            pad = np.zeros((N, F_target - F, T), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=1)

        # ── 3. Обрезка / паддинг по T ───────────────────────
        if T > T_target:
            spec = spec[:, :, :T_target]
        elif T < T_target:
            pad = np.zeros((N, F_target, T_target - T), dtype=spec.dtype)
            spec = np.concatenate([spec, pad], axis=2)

        return spec

    @staticmethod
    def _normalize(spec: np.ndarray) -> np.ndarray:
        """Min-max нормировка в [-1, 1] для каждого образца."""
        mn, mx = spec.min(), spec.max()
        if mx - mn > 1e-6:
            spec = 2.0 * (spec - mn) / (mx - mn) - 1.0
        return spec.astype(np.float32)
