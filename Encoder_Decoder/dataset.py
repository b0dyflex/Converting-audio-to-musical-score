"""
MidiSpectrogramDataset v2
=========================
Изменения по сравнению с v1:
  + skip_silent: фильтрация полностью пустых сегментов (спектрограмма ≈ 0)
  + silence_threshold: порог "пустоты" спектрограммы (по умолчанию 0.01)
  + Статистика по пустым сегментам при инициализации
  + Безопасная обработка: пустые сегменты не вызовут nan в loss

Каждый sample содержит:
  spectrogram.npy  --> (n_mels, time_steps)   float32   [0, 1]
  tokens.npy       --> (max_seq_len,)          int64

Dataset возвращает кортеж:
  spec       : FloatTensor (1, F, T)  — один сегмент
  src_tokens : LongTensor  (seq_len,) — [BOS, t1, t2, ..., tN]
  tgt_tokens : LongTensor  (seq_len,) — [t1,  t2, ..., tN, EOS]
  pad_mask   : BoolTensor  (seq_len,) — True там где PAD
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


class MidiSpectrogramDataset(Dataset):

    def __init__(
            self,
            dataset_root: str | Path,
            max_seq_len: int = 256,
            max_freq_bins: int = 128,
            max_time_steps: int = 216,
            skip_silent: bool = False,
            silence_threshold: float = 0.01,
            # Совместимость со старым train.py
            imagenet_norm: bool = False,
            max_segments: int = 0,
    ):
        self.root = Path(dataset_root)
        self.max_seq_len = max_seq_len
        self.max_freq_bins = max_freq_bins
        self.max_time_steps = max_time_steps
        self.silence_threshold = silence_threshold

        all_samples = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
               and (d / "spectrogram.npy").exists()
               and (d / "tokens.npy").exists()
        ])

        if not all_samples:
            raise RuntimeError(
                f"Нет данных в {self.root}.\n"
                "Запустите prepare_dataset.py для создания датасета."
            )

        # Проверка формата
        probe = np.load(all_samples[0] / "spectrogram.npy", mmap_mode="r")
        if probe.ndim == 3:
            raise RuntimeError(
                "Датасет создан старой версией prepare_dataset.py.\n"
                "Пересоздайте датасет."
            )

        # ── Фильтрация пустых сегментов ──────────────────────
        if skip_silent:
            self.samples = []
            n_skipped = 0
            for d in all_samples:
                spec = np.load(d / "spectrogram.npy", mmap_mode="r")
                # Считаем сегмент "пустым" если средняя энергия < порога
                if spec.mean() < silence_threshold:
                    n_skipped += 1
                else:
                    self.samples.append(d)
            print(f"Dataset v2: {len(self.samples)} сегментов "
                  f"(пропущено {n_skipped} пустых, порог={silence_threshold})")
        else:
            self.samples = all_samples
            print(f"Dataset v2: {len(self.samples)} сегментов (без фильтрации)")

        print(f"  Форма спектрограммы: (1, {max_freq_bins}, {max_time_steps})")
        print(f"  Нормировка: [0, 1] (dB-шкала)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        d = self.samples[idx]

        # ── Спектрограмма: (F, T) float32, значения [0, 1] ───
        spec = np.load(d / "spectrogram.npy")
        spec = self._fit_spec(spec)  # (F, T)
        spec_t = torch.from_numpy(spec).unsqueeze(0)  # (1, F, T)

        # ── Токены ────────────────────────────────────────────
        tokens = np.load(d / "tokens.npy").tolist()
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

        src_t = torch.tensor(src, dtype=torch.long)
        tgt_t = torch.tensor(tgt, dtype=torch.long)
        pad_mask = (src_t == PAD_TOKEN)

        return spec_t, src_t, tgt_t, pad_mask

    def _fit_spec(self, spec: np.ndarray) -> np.ndarray:
        """Приводит (F, T) к (max_freq_bins, max_time_steps)."""
        F, T = spec.shape
        Ft, Tt = self.max_freq_bins, self.max_time_steps
        if F > Ft:
            spec = spec[:Ft, :]
        elif F < Ft:
            spec = np.concatenate(
                [spec, np.zeros((Ft - F, T), dtype=spec.dtype)], axis=0)
        F = spec.shape[0]
        if T > Tt:
            spec = spec[:, :Tt]
        elif T < Tt:
            spec = np.concatenate(
                [spec, np.zeros((F, Tt - T), dtype=spec.dtype)], axis=1)
        return spec
