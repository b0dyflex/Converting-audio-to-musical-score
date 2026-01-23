import numpy as np
import matplotlib.pyplot as plt

fft = np.load("./midi_to_fft/data/2_fft.npy")

print("Shape:", fft.shape)
print("Min / Max:", fft.min(), fft.max())

plt.imshow(fft, aspect="auto", origin="lower")
plt.colorbar()
plt.title("FFT / Mel Spectrogram")
plt.show()
