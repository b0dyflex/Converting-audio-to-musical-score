"""
SpectrogramProcessor
====================
Преобразует аудио-сигнал в набор Mel-спектрограмм (один .npy файл = один трек).

Выход: np.ndarray формы (num_segments, n_mels, time_steps), dtype=float32

Параметры по умолчанию (AudioConfig):
  sample_rate  = 22050 Гц
  n_fft        = 2048  (~93 мс)
  hop_length   = 512   (~23 мс)
  win_length   = 1024  (~46 мс)
  n_mels       = 128
  fmax         = 8000 Гц
  segment_sec  = 5.0 с → 216 фреймов на сегмент при hop=512

Почему Mel, а не сырой FFT:
  - Mel-шкала логарифмична по частоте, как слух человека
  - Каждая октава занимает одинаковую полосу, ноты равномерно распределены
  - 128 Mel-бин охватывают 0–8000 Гц с хорошим разрешением внизу
  - Сырой FFT с n_fft=4096 давал 2049 бин, из которых используется лишь 128
    (линейная нижняя часть) — большинство нотных различий теряется
"""

import numpy as np
import librosa
from config import AudioConfig

# Константа: диапазон dB для нормировки (стандарт для музыки)
_DB_MIN = -80.0  # librosa power_to_db с top_db=80 даёт нижний предел −80 дБ
_DB_MAX = 0.0  # максимум всегда 0 дБ (ref=np.max)


class SpectrogramProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Разбивает аудио на сегменты и строит Mel-спектрограмму для каждого.

        Возвращает:
            np.ndarray формы (num_segments, n_mels, time_steps), float32
            Значения нормированы в диапазон [0, 1] по дБ-шкале.
            0.0 = тишина (−80 дБ), 1.0 = максимальная амплитуда (0 дБ)
        """
        cfg = self.config
        sr = cfg.sample_rate

        # ── 1. Параметры сегмента ─────────────────────────────
        segment_len = int(sr * cfg.segment_size_sec)
        if segment_len <= 0:
            raise ValueError("segment_size_sec слишком мал")

        # ── 2. Padding до кратного segment_len ───────────────
        remainder = len(audio) % segment_len
        if remainder != 0:
            audio = np.pad(audio, (0, segment_len - remainder))

        num_segments = len(audio) // segment_len
        all_specs = []

        # ── 3. Mel-спектрограмма для каждого сегмента ────────
        for i in range(num_segments):
            seg = audio[i * segment_len: (i + 1) * segment_len]

            if cfg.use_mel:
                # Mel-спектрограмма через librosa
                mel = librosa.feature.melspectrogram(
                    y=seg,
                    sr=sr,
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    win_length=cfg.win_length,
                    n_mels=cfg.n_mels,
                    fmax=cfg.fmax,
                    center=True,
                )
                if cfg.log_scale:
                    spec = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
                else:
                    spec = mel
            else:
                # Сырой FFT
                stft = librosa.stft(
                    seg,
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    win_length=cfg.win_length,
                    center=True,
                )
                spec = np.abs(stft)
                if cfg.log_scale:
                    spec = librosa.power_to_db(spec ** 2, ref=np.max, top_db=80.0)

            all_specs.append(spec)

        if not all_specs:
            raise ValueError("Не удалось сформировать ни одного сегмента")

        # ── 4. Стек и нормировка в [0, 1] ─────────────────────
        # (num_segments, n_mels/freq_bins, time_steps)
        spectrograms = np.stack(all_specs, axis=0).astype(np.float32)

        if cfg.log_scale:
            # dB диапазон: [−80, 0] → нормируем в [0, 1]
            # Это устойчивая нормировка: диапазон одинаков для всех сэмплов
            spectrograms = (spectrograms - _DB_MIN) / (_DB_MAX - _DB_MIN)
            spectrograms = np.clip(spectrograms, 0.0, 1.0)

        return spectrograms
