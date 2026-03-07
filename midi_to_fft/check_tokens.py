import numpy as np

try:
    tokens = np.load('D:/token_dataset/sample_0004/tokens.npy')

    total_tokens = tokens.size

    print(f"Форма массива: {tokens.shape}")
    print(f"Размерность: {tokens.ndim}")
    print(f"Тип данных: {tokens.dtype}")
    print(f"Общее количество токенов: {total_tokens}")


    if tokens.ndim > 1:
        print(f"\nДетали по измерениям:")
        for i, dim in enumerate(tokens.shape):
            print(f"  Измерение {i + 1}: {dim} элементов")
        print(f"\nПроверка: {tokens.shape} -> всего {total_tokens} токенов")

    print(f"\nПервые 10 токенов: {tokens.flatten()[:10]}")

    # Уникальные токены
    unique_tokens = np.unique(tokens)
    print(f"\nУникальных токенов: {len(unique_tokens)}")

except FileNotFoundError:
    print("Ошибка: Файл 'tokens.npy' не найден в текущей директории.")
except Exception as e:
    print(f"Произошла ошибка: {e}")