"""
ScoreGenerationModel
====================
Архитектура (см. схему):

  [Аудио] → SpectrogramProcessor → (N, F, T)
      ↓
  ConvEncoder (ResNet18)  →  Тензор [N, d_model]
      ↓ (память энкодера)
  TransformerDecoder
    └─ Output Embedding + PositionalEncoding
    └─ Masked Multi-Head Attention  (Add&Norm)
    └─ Multi-Head Attention         (Add&Norm)  ← cross-attention с энкодером
    └─ Feed Forward                 (Add&Norm)
    └─ Linear → Softmax → Output Probabilities
      ↓
  Последовательность токенов → Сгенерированная партитура

Параметры по умолчанию:
  d_model      = 128   (размер эмбеддинга, совпадает с выходом энкодера)
  nhead        = 8
  num_decoder_layers = 4
  dim_feedforward    = 512
  vocab_size   = 391   (из tokenizer.py)
  max_seq_len  = 512
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torchvision.models as tv_models

from tokenizer import VOCAB_SIZE, PAD_TOKEN


# ══════════════════════════════════════════════════════════════
#  1. Convolutional Encoder (ResNet18-based)
# ══════════════════════════════════════════════════════════════
class SpectrogramEncoder(nn.Module):
    """
    Принимает батч спектрограмм:
        x: (batch, N_segments, freq_bins, time_steps)
           например (B, 8, 128, 399)

    Возвращает:
        memory: (batch, N_segments, d_model)
                каждый сегмент закодирован в вектор d_model=128
    """

    def __init__(self, d_model: int = 128, pretrained: bool = False):
        super().__init__()
        self.d_model = d_model

        # ── Загрузка ResNet18 ────────────────────────────────
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Заменяем первый Conv2d: 3 channels → 1 channel
        backbone.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Убираем финальный классификатор (FC 512→1000)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # → (batch*N, 512, 1, 1)
        )

        # Проецируем 512 → d_model
        self.proj = nn.Sequential(
            nn.Flatten(),                    # (batch*N, 512)
            nn.Linear(512, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, F, T)
        """
        B, N, F, T = x.shape
        # Объединяем batch и сегменты
        x = x.view(B * N, 1, F, T)           # (B*N, 1, F, T)
        x = self.backbone(x)                  # (B*N, 512, 1, 1)
        x = self.proj(x)                      # (B*N, d_model)
        x = x.view(B, N, self.d_model)        # (B, N, d_model)
        return x


# ══════════════════════════════════════════════════════════════
#  2. Positional Encoding
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════
#  3. Transformer Decoder
# ══════════════════════════════════════════════════════════════
class MusicTransformerDecoder(nn.Module):
    """
    Стандартный Transformer-декодер согласно схеме:
      Output Embedding + PositionalEncoding
      → N × (MaskedMHA + AddNorm → CrossMHA + AddNorm → FFN + AddNorm)
      → Linear → Softmax
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding токенов выхода
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Стек декодерных слоёв PyTorch
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, seq, d_model)
            norm_first=False,   # Post-LN, как в оригинальном Transformer
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Выходной линейный слой
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        tgt: torch.Tensor,          # (B, tgt_len)   — целевые токены (shifted right)
        memory: torch.Tensor,       # (B, N, d_model) — выход энкодера
        tgt_key_padding_mask: torch.Tensor | None = None,   # (B, tgt_len)
        memory_key_padding_mask: torch.Tensor | None = None,# (B, N)
    ) -> torch.Tensor:
        """
        Возвращает логиты: (B, tgt_len, vocab_size)
        """
        tgt_len = tgt.size(1)

        # Каузальная маска (нижнетреугольная) для Masked MHA
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device
        )

        # Embedding + PositionalEncoding
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)   # (B, tgt_len, d_model)
        x = self.pos_encoding(x)

        # Декодер
        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_proj(x)   # (B, tgt_len, vocab_size)
        return logits


# ══════════════════════════════════════════════════════════════
#  4. Полная модель
# ══════════════════════════════════════════════════════════════
class ScoreGenerationModel(nn.Module):
    """
    Объединяет энкодер спектрограмм и трансформер-декодер.

    Использование:
        model = ScoreGenerationModel()

        # Прямой проход (обучение):
        logits = model(spectrograms, tgt_tokens)
        # logits: (B, tgt_len, vocab_size)

        # Генерация:
        tokens = model.generate(spectrograms, max_len=512)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pretrained_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = SpectrogramEncoder(d_model=d_model, pretrained=pretrained_encoder)
        self.decoder = MusicTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        print(f"ScoreGenerationModel | params: {self.count_parameters():,}")

    # ── Прямой проход ────────────────────────────────────────
    def forward(
        self,
        spectrograms: torch.Tensor,       # (B, N, F, T)
        tgt: torch.Tensor,                # (B, tgt_len)
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Возвращает логиты (B, tgt_len, vocab_size).
        Во время обучения tgt — tokens[:-1] (shifted right),
        цель — tokens[1:].
        """
        memory = self.encoder(spectrograms)    # (B, N, d_model)
        logits = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits

    # ── Автрегрессивная генерация ─────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        spectrograms: torch.Tensor,  # (1, N, F, T)  — один пример
        max_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Генерирует последовательность токенов авторегрессивно.
        Возвращает тензор (1, seq_len).
        """
        from tokenizer import BOS_TOKEN, EOS_TOKEN

        self.eval()
        device = spectrograms.device
        memory = self.encoder(spectrograms)    # (1, N, d_model)

        generated = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decoder(tgt=generated, memory=memory)
            next_logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            # Top-K фильтрация
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                threshold = values[:, -1].unsqueeze(1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == EOS_TOKEN:
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
