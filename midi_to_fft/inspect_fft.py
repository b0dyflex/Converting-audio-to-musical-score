import numpy as np
import matplotlib.pyplot as plt

specs = np.load("D:/token_dataset/sample_0001/tokens.npy")
print("Полная размерность тензора:", specs.shape)

num_segments = specs.shape[0]

fig, axes = plt.subplots(1, num_segments, figsize=(15, 4))

for i in range(num_segments):
    axes[i].imshow(specs[i], aspect="auto", origin="lower")
    axes[i].set_title(f"Seg {i}")

plt.tight_layout()
plt.show()
