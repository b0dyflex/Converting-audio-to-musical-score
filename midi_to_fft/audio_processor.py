import numpy as np
import librosa
from config import AudioConfig


class SpectrogramProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config

    def compute(self, audio: np.ndarray) -> np.ndarray:
        if self.config.use_mel:
            spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels
            )
        else:
            spectrogram = np.abs(
                librosa.stft(
                    audio,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.win_length
                )
            )

        if self.config.log_scale:
            spectrogram = librosa.power_to_db(
                spectrogram,
                ref=np.max
            )

        return spectrogram.astype(np.float32)
