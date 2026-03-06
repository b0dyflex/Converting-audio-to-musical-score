import numpy as np

tokens = np.load('D:/token_dataset/sample_0002/tokens.npy')
unique_tokens = np.unique(tokens)

print(f"Всего уникальных токенов: {len(unique_tokens)}")
print("Уникальные токены:")
print(unique_tokens)