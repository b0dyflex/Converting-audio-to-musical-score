"""
midi_to_fft

Подсистема генерации обучающих данных:
MIDI → WAV → FFT / Spectrogram

Используется для подготовки входных данных
для моделей машинного обучения.
"""

from .config import AudioConfig
from .midi_renderer import MidiRenderer
from .audio_processor import SpectrogramProcessor
from .fft_map_generator import MidiToFFTMap

__all__ = [
    "AudioConfig",
    "MidiRenderer",
    "SpectrogramProcessor",
    "MidiToFFTMap",
]
