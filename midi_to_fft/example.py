from config import AudioConfig
from fft_map_generator import MidiToFFTMap

config = AudioConfig()

pipeline = MidiToFFTMap(
    soundfont_path="./midi_to_fft/soundfonts/FluidR3_GM.sf2",
    config=config
)

fft_map = pipeline.process(
    midi_path="./midi_to_fft/examples/2.mid",
    output_wav="./midi_to_fft/data/2.wav",
    output_fft="./midi_to_fft/data/2_fft.npy"
)

print("FFT map shape:", fft_map.shape)
