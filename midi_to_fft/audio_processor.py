import numpy as np
import librosa
from config import AudioConfig


class SpectrogramProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Разбивает аудио на временные сегменты фиксированной длины.
        Для каждого сегмента строится спектрограмма (STFT).

        Возвращает:
            np.ndarray формы:
            (num_segments, freq_bins, time_steps)
        """

        sr = self.config.sample_rate

        # ===== 1. Параметры сегмента =====
        segment_size = int(sr * self.config.segment_size_sec)
        if segment_size <= 0:
            raise ValueError("segment_size_sec слишком мал")

        # ===== 2. Параметры STFT =====
        window_size = int(sr * self.config.window_size_ms / 1000)
        overlap = np.clip(self.config.overlap, 0.0, 0.99)
        hop = int(window_size * (1 - overlap))

        window = np.hanning(window_size)

        # ===== 3. Padding всего трека до кратности сегменту =====
        total_len = len(audio)
        remainder = total_len % segment_size

        if remainder != 0:
            pad_width = segment_size - remainder
            audio = np.pad(audio, (0, pad_width))

        num_segments = len(audio) // segment_size

        all_specs = []

        # ===== 4. Обработка каждого сегмента =====
        for seg_idx in range(num_segments):

            start = seg_idx * segment_size
            end = start + segment_size
            segment_audio = audio[start:end]

            frames = []

            # ---- STFT внутри сегмента ----
            for frame_start in range(0, len(segment_audio) - window_size + 1, hop):
                frame = segment_audio[frame_start:frame_start + window_size]
                frame = frame * window

                spectrum = np.fft.rfft(frame, n=self.config.n_fft)
                magnitude = np.abs(spectrum)

                frames.append(magnitude)

            if len(frames) == 0:
                continue

            spec = np.array(frames).T  # (freq_bins, time_steps)

            # ---- Mel-проекция ----
            if self.config.use_mel:
                mel_basis = librosa.filters.mel(
                    sr=sr,
                    n_fft=self.config.n_fft,
                    n_mels=self.config.n_mels
                )
                spec = np.dot(mel_basis, spec)

            # ---- Логарифмирование ----
            if self.config.log_scale:
                spec = librosa.power_to_db(spec ** 2, ref=np.max)

            all_specs.append(spec)

        if len(all_specs) == 0:
            raise ValueError("Не удалось сформировать сегменты")

        # (num_segments, freq_bins, time_steps)
        spectrograms = np.stack(all_specs, axis=0)

        return spectrograms.astype(np.float32)

