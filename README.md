# Converting-audio-to-musical-score
| Файл | роль |
|:---|:--- |
| tokenizer.py | Токенизатор MIDI в Токены и обратно |
| model.py | ResNet18 энкодер + Transformer декодер |
| dataset.py | PyTorch Dataset |
| prepare_dataset.py | Генерация датасета из MIDI-папки |
| train.py | Тренировочный пайплайн |
| inference.py | Генерация из аудиофайла |

<ul>Архитектура в деталях
  <li>Токенизатор (словарь 391 токен):
    <ul>
      <li>NOTE_ON/OFF × 128 нот, TIME_SHIFT × 100 шагов по 10 мс, VELOCITY × 32 бина + PAD/BOS/EOS</li>
    </ul>
  </li>
  <li>SpectrogramEncoder (ResNet18-based):
    <ul>
      <li>Входной тензор (B, N, 128, 399) разворачивается в (B×N, 1, 128, 399) — каждый сегмент как одноканальная «картинка»</li>
      <li>ResNet18 без первого слоя на 3 канала (заменён на 1-канальный) → avgpool → Linear(512→128) → LayerNorm</li>
      <li>Выход: (B, N, 128) — это память для декодера</li>
    </ul>
  </li>
  <li>TransformerDecoder (4 слоя)
    <ul>
      <li>Output Embedding + синусоидальный PositionalEncoding</li>
      <li>Каждый слой: Masked MHA → Add&Norm → Cross-MHA (key/value из encoder) → Add&Norm → FFN → Add&Norm</li>
      <li>Linear(128 → 391) → Softmax</li>
    </ul>
  </li>
  <li>Обучение: CrossEntropyLoss с label_smoothing=0.1, AdamW, warmup + cosine LR, mixed precision (torch.autocast).</li>
</ul>

Как запускать в Collab:
1. Подготовка датасета
<code>'!python prepare_dataset.py --midi_dir ./midi --output_dir ./dataset --soundfont FluidR3_GM.sf2'</code>

2. Обучение
<code>'!python train.py --dataset_dir ./dataset --output_dir ./checkpoints'</code>

3. Инференс
<code>'!python inference.py --audio song.wav --model_pt checkpoints/best_model.pt --output result.mid'</code>
