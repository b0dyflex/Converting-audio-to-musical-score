from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048

    n_mels: int = 128
    use_mel: bool = True
    log_scale: bool = True
