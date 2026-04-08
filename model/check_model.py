import torch

MODEL_PATH = r"C:\Users\badim\PycharmProjects\Converting-audio-to-musical-score\model\audio2music_best.pth"

checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print("=" * 60)
print("ИНФОРМАЦИЯ О МОДЕЛИ")
print("=" * 60)

print(f"\n1. Vocab size: {checkpoint['vocab_size']}")
print(f"2. Hidden dim (из conv_reduce): {checkpoint['model_state_dict']['encoder.conv_reduce.bias'].shape[0]}")

# Проверяем dim_feedforward по весам
for key in checkpoint['model_state_dict'].keys():
    if 'linear1.weight' in key:
        shape = checkpoint['model_state_dict'][key].shape
        print(f"3. {key}: shape={shape}")
        # shape[0] - это dim_feedforward
        if shape[0] == 2048:
            print(f"   → dim_feedforward = {shape[0]}, hidden_dim={shape[1]} → множитель = {shape[0] // shape[1]}")
        break

print(f"\n4. Ключи в чекпоинте (первые 20):")
for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:20]):
    print(f"   {key}")

print(f"\n5. Количество слоёв энкодера: {sum(1 for k in checkpoint['model_state_dict'].keys() if 'encoder.transformer_encoder.layers.' in k and '.norm1.weight' in k)}")
print(f"6. Количество слоёв декодера: {sum(1 for k in checkpoint['model_state_dict'].keys() if 'decoder.transformer_decoder.layers.' in k and '.norm1.weight' in k)}")