import torch
import sys
from pathlib import Path

ckpt_path = 'D:/model/Lesha/4/best_model.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

# last.pt — это словарь чекпоинта, а не сама модель
if isinstance(ckpt, dict):
    print("Ключи чекпоинта:", list(ckpt.keys()))
    if 'epoch' in ckpt:
        print(f"Эпоха: {ckpt['epoch']}")
    if 'val_loss' in ckpt:
        print(f"Val loss: {ckpt['val_loss']:.4f}")
    if 'best_val_loss' in ckpt:
        print(f"Best val loss: {ckpt['best_val_loss']:.4f}")
    if 'model' in ckpt:
        print(f"\nКлючи state_dict модели ({len(ckpt['model'])} слоёв):")
        for k, v in list(ckpt['model'].items())[:10]:
            print(f"  {k}: {tuple(v.shape)}")
        print("  ...")
else:
    # Если это сама модель (best_model.pt сохраняется как state_dict)
    print(ckpt)