from config import AudioConfig
from fft_map_generator import MidiToFFTMap

config = AudioConfig()

pipeline = MidiToFFTMap(
    soundfont_path="soundfonts/FluidR3_GM.sf2",
    config=config
)

fft_map = pipeline.process(
    midi_path="examples/1.mid",
    output_wav="data/1.wav",
    output_fft="data/1_fft.npy"
)

print("FFT map shape:", fft_map.shape)
