import torch

# Загрузить модель из файла PT
model = torch.load('D:/model/last.pt')

# Распечатать архитектуру модели
print(model)
