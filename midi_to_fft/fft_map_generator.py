import numpy as np
from config import AudioConfig
from midi_renderer import MidiRenderer
from audio_processor import SpectrogramProcessor


class MidiToFFTMap:
    def __init__(
        self,
        soundfont_path: str,
        config: AudioConfig
    ):
        self.renderer = MidiRenderer(
            soundfont_path,
            config.sample_rate
        )
        self.processor = SpectrogramProcessor(config)

    def process(
        self,
        midi_path: str,
        output_wav: str | None = None,
        output_fft: str | None = None
    ) -> np.ndarray:

        audio = self.renderer.render(
            midi_path,
            output_wav=output_wav
        )

        fft_map = self.processor.compute(audio)

        if output_fft:
            np.save(output_fft, fft_map)

        return fft_map
