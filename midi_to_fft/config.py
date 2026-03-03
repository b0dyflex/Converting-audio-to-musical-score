from dataclasses import dataclass


# @dataclass
# class AudioConfig:
#     sample_rate: int = 22050
#     n_fft: int = 2048
#     hop_length: int = 512
#     win_length: int = 2048
#     n_mels: int = 128
#     use_mel: bool = True
#     log_scale: bool = True

from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 22050

    # параметры окна
    window_size_ms: float = 25.0  # размер окна в миллисекундах
    overlap: float = 0.5  # доля перекрытия (0.0–0.9)
    segment_size_sec: float = 5.0  # длина одного сегмента

    # FFT
    n_fft: int = 4096
    use_mel: bool = False
    n_mels: int = 128
    log_scale: bool = True