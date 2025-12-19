import os
import pretty_midi
import soundfile as sf
import numpy as np


class MidiRenderer:
    def __init__(self, soundfont_path: str, sample_rate: int):
        if not os.path.exists(soundfont_path):
            raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")
        self.soundfont_path = soundfont_path
        self.sample_rate = sample_rate

    def render(
        self,
        midi_path: str,
        output_wav: str | None = None
    ) -> np.ndarray:
        midi = pretty_midi.PrettyMIDI(midi_path)
        audio = midi.fluidsynth(
            fs=self.sample_rate,
            sf2_path=self.soundfont_path
        )

        if output_wav:
            sf.write(output_wav, audio, self.sample_rate)

        return audio
