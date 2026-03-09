import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import time
import glob

warnings.filterwarnings('ignore')

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


# ==================== Residual Block ====================

class ResidualBlock(nn.Module):
    """Residual блок для энкодера"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


# ==================== AudioFeatureExtractor ====================

class AudioFeatureExtractor(nn.Module):
    """Оптический энкодер для извлечения признаков из аудио"""

    def __init__(self, input_channels=1, hidden_dim=512, num_layers=4, dropout=0.1):
        super().__init__()

        # Первоначальная свертка для увеличения каналов
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual блоки
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        ])

        # Адаптивная пуллинг и выравнивание
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.conv_reduce = nn.Conv1d(512, hidden_dim, kernel_size=1)

        # Трансформерный энкодер для временных зависимостей
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch, 1, freq_bins, time_steps]

        # Начальная свертка
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual блоки
        for block in self.res_blocks:
            x = block(x)

        # Адаптивный пулинг и подготовка для трансформера
        x = self.adaptive_pool(x).squeeze(2)  # [batch, 512, time_steps]
        x = self.conv_reduce(x)  # [batch, hidden_dim, time_steps]

        # Перестановка для трансформера
        x = x.permute(0, 2, 1)  # [batch, time_steps, hidden_dim]

        # Трансформер энкодер
        x = self.transformer_encoder(x)

        return x


# ==================== Music Tokenizer ====================

class MusicTokenizer:
    """Токенизатор для нотной партитуры"""

    def __init__(self, vocab_size=5000):
        self.vocab = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = vocab_size

        # Специальные токены для музыкальных событий
        self.pitch_tokens = {}
        self.duration_tokens = {}
        self.velocity_tokens = {}

    def build_from_midi_files(self, midi_files: List[str], max_vocab_size=5000):
        """Строит словарь из MIDI файлов"""
        print("Building vocabulary from MIDI files...")

        all_tokens = set()

        for midi_file in midi_files[:100]:  # Ограничиваем для скорости
            try:
                tokens = self._extract_tokens_from_midi(midi_file)
                all_tokens.update(tokens)
            except:
                continue

        # Добавляем токены в словарь
        for i, token in enumerate(sorted(all_tokens), start=len(self.vocab)):
            if i >= max_vocab_size:
                break
            self.vocab[token] = i

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary size: {self.vocab_size}")
        return self

    def _extract_tokens_from_midi(self, midi_path: str) -> List[str]:
        """Извлекает токены из MIDI файла"""
        tokens = []

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)

            # Извлекаем ноты
            for instrument in midi.instruments:
                for note in instrument.notes:
                    # Токен для начала ноты
                    pitch_token = f"PITCH_{note.pitch}"
                    velocity_token = f"VEL_{min(note.velocity // 10, 9)}"

                    # Токен для длительности (квантованная)
                    duration = note.end - note.start
                    quantized_duration = self._quantize_duration(duration)
                    duration_token = f"DUR_{quantized_duration}"

                    tokens.extend([pitch_token, velocity_token, duration_token])

                # Добавляем токены для пауз
                # (упрощённо - между концом одной ноты и началом следующей)

        except:
            pass

        return tokens

    def _quantize_duration(self, duration: float) -> str:
        """Квантует длительность"""
        # Базовые длительности в секундах (для темпа 120 BPM)
        durations = {
            0.125: '32nd',  # 32-я
            0.25: '16th',  # 16-я
            0.5: '8th',  # 8-я
            1.0: 'quarter',  # четверть
            2.0: 'half',  # половинная
            4.0: 'whole',  # целая
            8.0: 'double',  # бревис
        }

        # Находим ближайшую длительность
        best_match = 'quarter'  # по умолчанию
        min_diff = float('inf')

        for dur_value, dur_name in durations.items():
            diff = abs(duration - dur_value)
            if diff < min_diff:
                min_diff = diff
                best_match = dur_name

        return best_match

    def encode(self, tokens: List[str]) -> List[int]:
        """Кодирует токены в индексы"""
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Декодирует индексы в токены"""
        return [self.reverse_vocab.get(idx, '<UNK>') for idx in indices]

    def tokenize_midi(self, midi_path: str) -> List[int]:
        """Полный пайплайн токенизации MIDI"""
        tokens = self._extract_tokens_from_midi(midi_path)
        encoded = self.encode(['<SOS>'] + tokens + ['<EOS>'])
        return encoded


# ==================== Positional Encoding ====================

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ==================== Transformer Decoder ====================

class MusicTransformerDecoder(nn.Module):
    """Трансформер-декодер для генерации нотных последовательностей"""

    def __init__(self, vocab_size, hidden_dim=512, num_layers=6, dropout=0.1, max_len=500):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: [batch, tgt_len]
        # memory: [batch, src_len, hidden_dim] из энкодера

        tgt_emb = self.embedding(tgt) * np.sqrt(self.hidden_dim)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Создаём маску для декодера
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        """Создаёт маску для автогрессивного декодирования"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        return mask


# ==================== Audio2Music Model ====================

class Audio2MusicModel(nn.Module):
    """Полная модель для преобразования аудио в ноты"""

    def __init__(self, vocab_size, hidden_dim=512, num_encoder_layers=4, num_decoder_layers=6):
        super().__init__()

        self.encoder = AudioFeatureExtractor(
            input_channels=1,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers
        )

        self.decoder = MusicTransformerDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers
        )

    def forward(self, audio_features, target_tokens, target_mask=None):
        # audio_features: [batch, 1, freq_bins, time_steps]
        # target_tokens: [batch, seq_len]

        # Энкодируем аудио
        encoded_audio = self.encoder(audio_features)  # [batch, time_steps, hidden_dim]

        # Увеличиваем временное разрешение для лучшего выравнивания
        if encoded_audio.size(1) < 100:
            encoded_audio = F.interpolate(
                encoded_audio.transpose(1, 2),
                size=100,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Генерируем выход
        output = self.decoder(target_tokens, encoded_audio)  # [batch, seq_len-1, vocab_size]

        return output

    def generate(self, audio_features, max_len=500, temperature=1.0):
        """Генерация нотной последовательности из аудио"""
        self.eval()

        batch_size = audio_features.size(0)
        device = audio_features.device

        # Энкодируем аудио
        with torch.no_grad():
            encoded_audio = self.encoder(audio_features)

            if encoded_audio.size(1) < 100:
                encoded_audio = F.interpolate(
                    encoded_audio.transpose(1, 2),
                    size=100,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

        # Начинаем с SOS токена
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device) * 1  # SOS token

        for i in range(max_len):
            with torch.no_grad():
                output = self.decoder(generated, encoded_audio)

                # Берём последний токен
                next_token_logits = output[:, -1, :] / temperature

                # Применяем softmax и сэмплируем
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)

                generated = torch.cat([generated, next_token], dim=1)

                # Проверяем EOS токен
                if (next_token == 2).all():  # EOS token
                    break

        return generated


# ==================== Dataset ====================

class AudioMusicDataset(Dataset):
    """Датасет для обучения модели Audio2Music (GPU оптимизированный)"""

    def __init__(self, data_dir, tokenizer, sample_rate=22050, n_fft=2048,
                 hop_length=512, max_audio_len=10, generate_audio_from_midi=True,
                 max_samples=500):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_len = max_audio_len
        self.generate_audio_from_midi = generate_audio_from_midi
        self.max_samples = max_samples

        # Используем кэш для ускорения загрузки
        self.cache = {}
        self.use_cache = True

        print(f"Initializing GPU-optimized dataset from: {self.data_dir}")

        # Собираем пары аудио-MIDI
        self.pairs = self._collect_pairs()

        # Ограничиваем количество пар для быстрого прототипирования
        if len(self.pairs) > self.max_samples:
            print(f"Limiting dataset from {len(self.pairs)} to {self.max_samples} samples")
            self.pairs = self.pairs[:self.max_samples]

        print(f"Found {len(self.pairs)} audio-midi pairs")

    def _collect_pairs(self):
        """Собирает пары аудио-MIDI файлов"""
        pairs = []

        # Ищем MIDI файлы
        midi_extensions = ['.mid', '.midi']
        midi_files = []

        # Проверяем все возможные пути
        search_paths = [
            self.data_dir,
            self.data_dir / 'midi',
            self.data_dir / 'MIDI',
            self.data_dir / 'midi_files',
            self.data_dir / 'MIDI_Files'
        ]

        for search_path in search_paths:
            if search_path.exists():
                for ext in midi_extensions:
                    midi_files.extend(search_path.glob(f'**/*{ext}'))
                    midi_files.extend(search_path.glob(f'*{ext}'))

        # Убираем дубликаты
        midi_files = list(set(midi_files))

        print(f"Found {len(midi_files)} MIDI files in dataset")

        for midi_file in midi_files:
            if self.generate_audio_from_midi:
                # Генерируем аудио из MIDI на лету
                pairs.append((midi_file, midi_file))
            else:
                # Ищем существующее аудио
                audio_found = False
                audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']

                # Проверяем разные возможные расположения аудио файлов
                possible_audio_locations = [
                    midi_file.with_suffix(audio_ext) for audio_ext in audio_extensions
                ]

                # Также проверяем в папке audio/ если есть
                audio_dir = self.data_dir / 'audio'
                if audio_dir.exists():
                    for audio_ext in audio_extensions:
                        audio_name = midi_file.stem + audio_ext
                        possible_audio_locations.append(audio_dir / audio_name)

                for audio_file in possible_audio_locations:
                    if audio_file.exists():
                        pairs.append((audio_file, midi_file))
                        audio_found = True
                        break

        if not pairs:
            print("No audio-MIDI pairs found!")

        return pairs

    def _midi_to_audio(self, midi_path, max_duration=10):
        """Конвертирует MIDI в аудио с помощью pretty_midi"""
        try:
            # Загружаем MIDI
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            # Ограничиваем длительность
            total_duration = min(midi.get_end_time(), max_duration)

            # Генерируем аудио
            audio = midi.fluidsynth(fs=self.sample_rate)

            # Обрезаем до max_duration
            max_samples = int(max_duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            return audio

        except Exception as e:
            print(f"Error converting MIDI to audio {midi_path}: {e}")
            # Возвращаем тишину
            return np.zeros(int(max_duration * self.sample_rate))

    def __getitem__(self, idx):
        # Используем кэш если есть
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        audio_path_or_midi, midi_path = self.pairs[idx]

        try:
            # Загружаем и обрабатываем аудио
            if self.generate_audio_from_midi and str(audio_path_or_midi).lower().endswith(('.mid', '.midi')):
                # Генерируем аудио из MIDI
                audio = self._midi_to_audio(audio_path_or_midi, self.max_audio_len)
                sr = self.sample_rate
            else:
                # Загружаем существующее аудио
                audio, sr = librosa.load(str(audio_path_or_midi), sr=self.sample_rate,
                                         duration=self.max_audio_len)

            # Вычисляем спектрограмму
            spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=128
            )

            # Логарифмическая шкала и нормализация
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)

            # Преобразуем в тензор и паддим до фиксированной ширины
            # (разные WAV имеют разную длину → разные time_steps → батч не собирается)
            FIXED_TIME = 431  # ~10 сек при sr=22050, hop=512
            spectrogram = torch.FloatTensor(spectrogram)  # [n_mels, time]
            time_steps = spectrogram.shape[1]
            if time_steps >= FIXED_TIME:
                spectrogram = spectrogram[:, :FIXED_TIME]
            else:
                pad = torch.zeros(spectrogram.shape[0], FIXED_TIME - time_steps)
                spectrogram = torch.cat([spectrogram, pad], dim=1)
            spectrogram = spectrogram.unsqueeze(0)  # [1, n_mels, FIXED_TIME]

            # Токенизируем MIDI
            try:
                token_ids = self.tokenizer.tokenize_midi(str(midi_path))
            except:
                token_ids = [1, 2]  # SOS, EOS

            token_tensor = torch.LongTensor(token_ids)

            # Обрезаем или дополняем последовательность
            max_token_len = 500
            if len(token_tensor) > max_token_len:
                token_tensor = token_tensor[:max_token_len]
            else:
                padding = torch.zeros(max_token_len - len(token_tensor), dtype=torch.long)
                token_tensor = torch.cat([token_tensor, padding])

            result = (spectrogram, token_tensor)

            # Сохраняем в кэш
            if self.use_cache:
                self.cache[idx] = result

            return result

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Возвращаем нулевые тензоры фиксированного размера
            spectrogram = torch.zeros(1, 128, 431)  # [1, n_mels, FIXED_TIME]
            token_tensor = torch.zeros(500, dtype=torch.long)
            return (spectrogram, token_tensor)

    def __len__(self):
        return len(self.pairs)


# ==================== Trainer (GPU оптимизированный) ====================

class Audio2MusicTrainer:
    """Тренер для модели Audio2Music (GPU оптимизированный)"""

    def __init__(self, model, tokenizer, device=device):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Оптимизатор и loss функция
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)

        # Планировщик обучения
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def _save_checkpoint(self, path, epoch, train_loss, val_loss, is_best):
        """Сохранение чекпоинта модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'tokenizer': self.tokenizer,
            'vocab_size': self.tokenizer.vocab_size,
            'is_best': is_best,
            'device': str(self.device),
            'timestamp': time.time()
        }
        torch.save(checkpoint, path)
        return path

    def train_epoch(self, train_loader):
        """Одна эпоха обучения с GPU"""
        self.model.train()
        total_loss = 0
        processed_batches = 0

        for batch_idx, (audio_features, target_tokens) in enumerate(train_loader):
            # Очищаем кэш CUDA периодически
            if batch_idx % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Перемещаем данные на GPU
            audio_features = audio_features.to(self.device, non_blocking=True)
            target_tokens = target_tokens.to(self.device, non_blocking=True)

            # Проверяем размеры данных
            if audio_features.shape[0] == 0 or target_tokens.shape[0] == 0:
                continue

            # Подготовка входа и цели для учительского форсинга
            decoder_input = target_tokens[:, :-1]
            decoder_target = target_tokens[:, 1:]

            # Проверяем, что последовательности не пустые
            if decoder_input.shape[1] == 0 or decoder_target.shape[1] == 0:
                continue

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(audio_features, decoder_input)

            # Проверяем размеры
            if output.shape[0] != decoder_target.shape[0] or output.shape[1] != decoder_target.shape[1]:
                continue

            # Reshape для loss функции
            output = output.reshape(-1, output.size(-1))
            decoder_target = decoder_target.reshape(-1)

            # Вычисляем loss
            loss = self.criterion(output, decoder_target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            processed_batches += 1

            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}")

        if processed_batches == 0:
            return 0.0

        avg_loss = total_loss / processed_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, val_loader):
        """Валидация на GPU"""
        if val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        processed_batches = 0

        with torch.no_grad():
            for audio_features, target_tokens in val_loader:
                # Перемещаем данные на GPU
                audio_features = audio_features.to(self.device, non_blocking=True)
                target_tokens = target_tokens.to(self.device, non_blocking=True)

                # Пропускаем пустые батчи
                if audio_features.shape[0] == 0 or target_tokens.shape[0] == 0:
                    continue

                decoder_input = target_tokens[:, :-1]
                decoder_target = target_tokens[:, 1:]

                # Пропускаем пустые последовательности
                if decoder_input.shape[1] == 0 or decoder_target.shape[1] == 0:
                    continue

                output = self.model(audio_features, decoder_input)

                # Проверяем размеры
                if output.shape[0] != decoder_target.shape[0] or output.shape[1] != decoder_target.shape[1]:
                    continue

                output = output.reshape(-1, output.size(-1))
                decoder_target = decoder_target.reshape(-1)

                loss = self.criterion(output, decoder_target)
                total_loss += loss.item()
                processed_batches += 1

        if processed_batches == 0:
            return 0.0

        avg_loss = total_loss / processed_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self, train_loader, val_loader, epochs=20, save_dir='./models'):
        """Полный цикл обучения с GPU"""

        # Создаем директорию для сохранения
        os.makedirs(save_dir, exist_ok=True)

        # Основные пути сохранения
        best_path = os.path.join(save_dir, 'audio2music_best.pth')
        final_path = os.path.join(save_dir, 'audio2music_final.pth')

        print(f"\nStarting training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Models will be saved to: {save_dir}")

        total_start_time = time.time()
        saved_files = []  # Список сохраненных файлов

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 60}")

            # Очищаем кэш перед каждой эпохой
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Обучение
            train_loss = self.train_epoch(train_loader)
            train_time = time.time() - epoch_start_time

            # Валидация
            val_start_time = time.time()
            val_loss = self.validate(val_loader)
            val_time = time.time() - val_start_time

            # Общее время эпохи
            epoch_time = time.time() - epoch_start_time

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Time: {train_time:.1f}s)")
            print(f"  Val Loss:   {val_loss:.4f} (Time: {val_time:.1f}s)")
            print(f"  Epoch Time: {epoch_time:.1f}s")

            # Планировщик
            if val_loader:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)

            # Выводим текущий learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.2e}")

            # Сохранение лучшей модели
            is_best = False
            if val_loader:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    saved_path = self._save_checkpoint(best_path, epoch, train_loss, val_loss, is_best=True)
                    saved_files.append(saved_path)
                    print(f"  ✓ Saved BEST model to {saved_path}")
            else:
                is_best = train_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_loss
                    saved_path = self._save_checkpoint(best_path, epoch, train_loss, val_loss, is_best=True)
                    saved_files.append(saved_path)
                    print(f"  ✓ Saved BEST model to {saved_path}")

            # Сохраняем чекпоинт каждые 5 эпох
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'audio2music_epoch_{epoch + 1}.pth')
                saved_path = self._save_checkpoint(checkpoint_path, epoch, train_loss, val_loss, is_best=False)
                saved_files.append(saved_path)
                print(f"  ✓ Saved checkpoint to {saved_path}")

            # Ранняя остановка
            if len(self.val_losses) > 5 and val_loader:
                if min(self.val_losses[-5:]) > self.best_val_loss:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        total_time = time.time() - total_start_time

        # Сохраняем финальную модель
        saved_path = self._save_checkpoint(final_path, epochs - 1, train_loss, val_loss, is_best=False)
        saved_files.append(saved_path)

        print(f"\n{'=' * 60}")
        print(f"Training completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to {final_path}")

        # Выводим информацию о сохраненных файлах
        self._print_saved_files_info(saved_files, save_dir)

        return self.best_val_loss

    def _print_saved_files_info(self, saved_files, save_dir):
        """Выводит информацию о сохраненных файлах"""
        print(f"\n{'=' * 60}")
        print("СОХРАНЕННЫЕ ФАЙЛЫ МОДЕЛИ:")
        print(f"{'=' * 60}")

        current_dir = os.getcwd()
        print(f"Текущая рабочая папка: {current_dir}")
        print(f"Папка для моделей: {save_dir}")
        print()

        # Проверяем все файлы .pth в папке сохранения
        all_pth_files = glob.glob(os.path.join(save_dir, "*.pth"))

        if not all_pth_files:
            print("❌ Файлы модели не найдены!")
            return

        print(f"Найдено {len(all_pth_files)} файлов модели:")
        for file_path in sorted(all_pth_files):
            file_name = os.path.basename(file_path)
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # в MB
                # Пытаемся загрузить метаданные
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 'N/A')
                val_loss = checkpoint.get('val_loss', 'N/A')
                is_best = checkpoint.get('is_best', False)

                best_marker = " (BEST)" if is_best else ""
                print(f"  ✓ {file_name}{best_marker}")
                print(f"    Размер: {file_size:.1f} MB")
                print(f"    Эпоха: {epoch}, Val Loss: {val_loss:.4f}")
                print(f"    Полный путь: {os.path.abspath(file_path)}")
                print()

            except Exception as e:
                print(f"  ⚠ {file_name} - ошибка чтения: {e}")
                print(f"    Полный путь: {os.path.abspath(file_path)}")
                print()


# ==================== Data Preparation ====================

def prepare_training_data_gpu(data_dir='S:/Music Dataset', batch_size=4, num_workers=2, max_samples=200):
    """Подготовка данных для обучения с GPU оптимизацией"""

    print(f"\nPreparing data from: {data_dir}")

    # Проверяем существование директории
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Creating simulated data in ./data...")
        data_dir = './data'
        create_simulated_data(data_dir)

    # Находим MIDI файлы
    midi_patterns = [f'{data_dir}/**/*.mid', f'{data_dir}/**/*.midi']
    midi_files = []
    for pattern in midi_patterns:
        midi_files.extend(glob.glob(pattern, recursive=True))

    midi_files = list(set(midi_files))
    print(f"Found {len(midi_files)} MIDI files")

    if not midi_files:
        print("Creating simulated data...")
        create_simulated_data(data_dir)
        midi_files = glob.glob(f'{data_dir}/*.mid') + glob.glob(f'{data_dir}/*.midi')

    if not midi_files:
        raise ValueError("No MIDI files found!")

    # Создаём токенизатор
    print("Building tokenizer...")
    tokenizer = MusicTokenizer()
    tokenizer.build_from_midi_files(midi_files[:50])  # Используем первые 50 файлов

    # Создаём датасет
    print("Creating dataset...")
    dataset = AudioMusicDataset(
        data_dir,
        tokenizer,
        generate_audio_from_midi=True,
        max_samples=max_samples
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")

    print(f"Dataset created with {len(dataset)} samples")

    # Разделяем на train/val
    if len(dataset) < 10:
        # Маленький датасет
        train_size = len(dataset)
        val_size = 0
        train_dataset = dataset
        val_dataset = None
        batch_size = min(2, len(dataset))
        print(f"Small dataset: using all {train_size} samples for training")
    else:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        batch_size = min(batch_size, len(train_dataset))
        print(f"Train/Val split: {train_size}/{val_size}")

    # num_workers=0 — обязательно на Windows: при num_workers>0 каждый
    # worker-процесс импортирует model.py заново, что вызывает лишние
    # print'ы ("Using device: cuda") и проблемы с multiprocessing.
    num_workers = 0

    print(f"Using batch size: {batch_size}, num_workers: {num_workers}")

    # Создаём DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
        )
    else:
        val_loader = None

    return train_loader, val_loader, tokenizer


# ==================== Simulated Data ====================

def create_simulated_data(data_dir):
    """Создаёт симулированные данные для демонстрации"""
    import os
    from pathlib import Path

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True, parents=True)

    print(f"Creating simulated data in: {data_path}")

    # Создаём несколько MIDI файлов для разнообразия
    for file_num in range(3):
        midi = pretty_midi.PrettyMIDI()

        # Создаём инструмент
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # Добавляем несколько нот с разными мелодиями
        for i in range(8):
            note = pretty_midi.Note(
                velocity=80 + (i * 5),
                pitch=60 + (i % 5),  # Простая мелодия
                start=i * 0.5,
                end=i * 0.5 + 0.25 + (i % 3) * 0.1
            )
            piano.notes.append(note)

        midi.instruments.append(piano)

        # Сохраняем MIDI
        midi_path = data_path / f'sample_{file_num}.mid'
        midi.write(str(midi_path))

        # Создаём соответствующий аудио файл
        audio = midi.fluidsynth(fs=22050)

        import soundfile as sf
        audio_path = data_path / f'sample_{file_num}.wav'
        sf.write(str(audio_path), audio, 22050)

        print(f"  Created: {midi_path.name} and {audio_path.name}")

    print(f"Simulated data created successfully in {data_path}")


# ==================== Inference Pipeline ====================

def _infer_arch_from_checkpoint(checkpoint: dict) -> tuple:
    """
    Определяет архитектуру модели (hidden_dim, enc_layers, dec_layers)
    непосредственно из весов сохранённого чекпоинта.
    Это необходимо потому, что во время обучения размер модели выбирается
    автоматически в зависимости от доступной VRAM, и может отличаться
    от жёстко заданных дефолтных значений.
    """
    state = checkpoint['model_state_dict']

    # hidden_dim — размер bias вектора conv_reduce
    hidden_dim = state['encoder.conv_reduce.bias'].shape[0]

    # enc_layers — количество слоёв энкодера (считаем уникальные индексы)
    enc_layers = sum(
        1 for k in state
        if k.startswith('encoder.transformer_encoder.layers.')
        and k.endswith('.norm1.weight')
    )

    # dec_layers — количество слоёв декодера
    dec_layers = sum(
        1 for k in state
        if k.startswith('decoder.transformer_decoder.layers.')
        and k.endswith('.norm1.weight')
    )

    return hidden_dim, enc_layers, dec_layers


class Audio2MusicInference:
    """Класс для инференса модели"""

    def __init__(self, model_path=None, device=device):
        self.device = device

        # Если путь не указан, ищем модель
        if model_path is None:
            # Ищем модель в разных местах
            possible_paths = [
                './models/audio2music_best.pth',
                './models/audio2music_final.pth',
                './audio2music_best.pth',
                './audio2music_final.pth'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                raise FileNotFoundError("No model file found. Please train the model first.")

        print(f"Loading model from: {model_path}")

        # Загружаем чекпоинт
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.tokenizer = checkpoint['tokenizer']
        vocab_size = checkpoint['vocab_size']

        # Читаем архитектуру из чекпоинта — она могла отличаться от дефолтной
        # в зависимости от доступной VRAM во время обучения
        hidden_dim, enc_layers, dec_layers = _infer_arch_from_checkpoint(checkpoint)
        print(f"  Архитектура из чекпоинта: hidden_dim={hidden_dim}, "
              f"enc_layers={enc_layers}, dec_layers={dec_layers}")

        # Создаём и загружаем модель с правильными размерами
        self.model = Audio2MusicModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully. Vocab size: {vocab_size}")

    def transcribe_audio(self, audio_path, output_midi_path='output.mid',
                         output_musicxml_path='output.musicxml'):
        """Транскрибирует аудио в нотную партитуру"""

        # Загружаем и обрабатываем аудио
        audio, sr = librosa.load(audio_path, sr=22050)

        # Создаём спектрограмму
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)

        # Подготавливаем для модели
        audio_features = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)

        # Генерируем токены
        with torch.no_grad():
            generated_tokens = self.model.generate(audio_features)

        # Конвертируем токены в MIDI
        midi = self._tokens_to_midi(generated_tokens[0].cpu().numpy())
        midi.write(output_midi_path)

        # Конвертируем MIDI в MusicXML
        self._convert_to_musicxml(output_midi_path, output_musicxml_path)

        print(f"\nTranscription complete!")
        print(f"MIDI saved to: {output_midi_path}")
        print(f"MusicXML saved to: {output_musicxml_path}")

        return output_midi_path, output_musicxml_path

    def _tokens_to_midi(self, tokens):
        """Конвертирует токены в MIDI объект"""
        # Убираем специальные токены
        tokens = [t for t in tokens if t not in [0, 1, 2, 3]]  # PAD, SOS, EOS, UNK

        # Создаём MIDI
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0
        i = 0

        while i < len(tokens):
            if i < len(self.tokenizer.reverse_vocab):
                token_str = self.tokenizer.reverse_vocab.get(tokens[i], '')
                if token_str.startswith('PITCH_'):
                    # Извлекаем информацию о ноте
                    pitch = int(token_str.split('_')[1])

                    if i + 2 < len(tokens):
                        vel_token = self.tokenizer.reverse_vocab.get(tokens[i + 1], 'VEL_7')
                        dur_token = self.tokenizer.reverse_vocab.get(tokens[i + 2], 'DUR_quarter')

                        velocity = int(vel_token.split('_')[1]) * 10 + 20

                        # Конвертируем длительность в секунды
                        duration_map = {
                            '32nd': 0.125,
                            '16th': 0.25,
                            '8th': 0.5,
                            'quarter': 1.0,
                            'half': 2.0,
                            'whole': 4.0,
                            'double': 8.0
                        }
                        duration = duration_map.get(dur_token.split('_')[1], 1.0)

                        # Создаём ноту
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=current_time,
                            end=current_time + duration
                        )
                        piano.notes.append(note)

                        current_time += duration
                        i += 3
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1

        midi.instruments.append(piano)
        return midi

    def _convert_to_musicxml(self, midi_path, musicxml_path):
        """Конвертирует MIDI в MusicXML"""
        try:
            from music21 import converter
            score = converter.parse(midi_path)
            score.write('musicxml', fp=musicxml_path)
        except:
            print("Warning: Could not convert to MusicXML. Make sure music21 is installed.")
            # Создаём простой MusicXML файл
            self._create_simple_musicxml(musicxml_path)

    def _create_simple_musicxml(self, output_path):
        """Создаёт простой MusicXML файл для демонстрации"""
        simple_xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
                                "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>G</sign>
          <line>2</line>
        </clef>
      </attributes>
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simple_xml)


# ==================== Main Function ====================

def main():
    """Основная функция с GPU оптимизацией"""
    print("=" * 60)
    print("Audio to Music Transcription Model - GPU Optimized")
    print("=" * 60)

    # Проверка GPU
    if device.type == 'cuda':
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    else:
        print("\nWARNING: No GPU detected, using CPU (training will be slow!)")

    # Подготовка данных
    print("\n1. Preparing data with GPU optimization...")
    try:
        # Пробуем ваш путь к датасету
        data_path = r'S:\Music Dataset'
        print(f"Trying data path: {data_path}")

        train_loader, val_loader, tokenizer = prepare_training_data_gpu(
            data_dir=data_path,
            batch_size=8 if device.type == 'cuda' else 2,
            num_workers=4 if device.type == 'cuda' else 0,
            max_samples=200  # Ограничиваем для быстрого обучения
        )
    except Exception as e:
        print(f"Error loading data from S:\\Music Dataset: {e}")
        print("Trying with local data...")
        try:
            train_loader, val_loader, tokenizer = prepare_training_data_gpu(
                data_dir='./data',
                batch_size=4,
                num_workers=2,
                max_samples=100
            )
        except Exception as e2:
            print(f"Error with local data: {e2}")
            print("Creating demo data...")
            create_simulated_data('./data')
            train_loader, val_loader, tokenizer = prepare_training_data_gpu(
                data_dir='./data',
                batch_size=2,
                num_workers=0,
                max_samples=50
            )

    if len(train_loader.dataset) == 0:
        print("ERROR: No training data available!")
        return

    # Создание модели
    print("\n2. Creating model...")

    vocab_size = max(100, tokenizer.vocab_size)

    # Выбираем размер модели в зависимости от доступности GPU
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if gpu_memory_gb >= 8:
            hidden_dim = 512
            encoder_layers = 4
            decoder_layers = 6
            print(f"Large GPU ({gpu_memory_gb:.1f}GB): Using full model")
        elif gpu_memory_gb >= 4:
            hidden_dim = 384
            encoder_layers = 3
            decoder_layers = 4
            print(f"Medium GPU ({gpu_memory_gb:.1f}GB): Using medium model")
        else:
            hidden_dim = 256
            encoder_layers = 2
            decoder_layers = 3
            print(f"Small GPU ({gpu_memory_gb:.1f}GB): Using small model")
    else:
        hidden_dim = 256
        encoder_layers = 2
        decoder_layers = 3
        print("CPU: Using small model for faster training")

    model = Audio2MusicModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=encoder_layers,
        num_decoder_layers=decoder_layers
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Создание тренера
    trainer = Audio2MusicTrainer(model, tokenizer, device)

    # Определяем количество эпох
    dataset_size = len(train_loader.dataset)
    if dataset_size < 50:
        epochs = 10
    elif dataset_size < 200:
        epochs = 15
    else:
        epochs = 20

    print(f"\nDataset size: {dataset_size} samples")
    print(f"Training for {epochs} epochs")

    # Обучение
    print("\n3. Starting training...")

    # Создаем папку для моделей
    save_dir = './models'
    best_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir
    )

    # Демонстрация инференса
    print("\n" + "=" * 60)
    print("Testing Inference...")
    print("=" * 60)

    # Проверяем, есть ли сохраненные модели
    model_files = glob.glob(os.path.join(save_dir, "*.pth"))
    if not model_files:
        print("No model files found in ./models/")
        # Проверяем текущую папку
        model_files = glob.glob("*.pth")

    if model_files:
        print(f"Found {len(model_files)} model file(s)")

        # Используем лучшую модель
        best_model = None
        for model_file in model_files:
            if 'best' in model_file.lower():
                best_model = model_file
                break

        if best_model is None:
            best_model = model_files[0]  # Используем первый найденный

        print(f"Using model: {best_model}")

        try:
            # Создаём инференс
            inferencer = Audio2MusicInference(best_model, device=device)
            print("✓ Inference pipeline created")

            # Тестируем на примере
            print("\nTesting transcription on sample audio...")

            # Ищем тестовое аудио
            test_audio = None
            possible_audio_paths = [
                r'S:\Music Dataset\*.wav',
                './data/*.wav',
                './*.wav'
            ]

            for pattern in possible_audio_paths:
                files = glob.glob(pattern)
                if files:
                    test_audio = files[0]
                    break

            if test_audio and os.path.exists(test_audio):
                print(f"Found test audio: {test_audio}")

                # Транскрибируем
                midi_path, musicxml_path = inferencer.transcribe_audio(
                    test_audio,
                    output_midi_path='test_output.mid',
                    output_musicxml_path='test_output.musicxml'
                )
                print(f"\n✓ Test completed successfully!")
                print(f"Output saved to: {midi_path}")
            else:
                print("No audio files found for testing.")
                print("You can test the model later with:")
                print(f"  inferencer = Audio2MusicInference('{best_model}')")
                print("  inferencer.transcribe_audio('your_audio.wav', 'output.mid')")

        except Exception as e:
            print(f"✗ Error during inference test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ No trained model files found!")
        print("Model training may have failed or models were not saved.")

    print("\n" + "=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)

    # Выводим итоговую информацию
    print(f"\nSummary:")
    print(f"- Device used: {device}")
    print(f"- Training time: ~{epochs} epochs")
    print(f"- Models saved to: {save_dir}/")
    print(f"- Best validation loss: {best_loss:.4f}")

    if device.type == 'cuda':
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")


# ==================== Запуск ====================

if __name__ == "__main__":
    # Запускаем основную функцию
    main()

    # Инструкция для последующего использования
    # print("\n" + "=" * 60)
    # print("HOW TO USE THE TRAINED MODEL:")
    # print("=" * 60)
    # print("\n1. For inference with a trained model:")
    # print("   from model import Audio2MusicInference")
    # print("   inferencer = Audio2MusicInference('./models/audio2music_best.pth')")
    # print("   inferencer.transcribe_audio('your_audio.wav', 'output.mid')")
    # print("\n2. To retrain the model:")
    # print("   from model import main")
    # print("   main()")
    # print("\n3. Check the 'models/' folder for saved model files.")