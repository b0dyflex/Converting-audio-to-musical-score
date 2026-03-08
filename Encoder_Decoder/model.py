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

"""
ScoreGenerationModel  v2  (оптимизировано под 8 ГБ GPU)
=========================================================

Улучшения по сравнению с v1:

1. ЭНКОДЕР — лёгкий кастомный CNN вместо ResNet18/EfficientNet
   - 4 блока Conv→BN→GELU с постепенным увеличением каналов (32→64→128→256)
   - Depthwise separable convolutions в последних блоках — экономия памяти
   - Потребление памяти в ~4× меньше, чем EfficientNet-B0

2. ЭНКОДЕР — Positional Encoding для сегментов
   - Модель знает порядок временных сегментов

3. ЭНКОДЕР — 1-слойный Transformer Encoder поверх CNN
   - Сегменты видят контекст друг друга через self-attention

4. ДЕКОДЕР — Pre-LN (norm_first=True) вместо Post-LN
   - Стабилизирует градиенты, быстрее сходится

5. ДЕКОДЕР — Weight Tying (embedding ↔ output projection)
   - Снижает переобучение, стандартная техника GPT/T5

6. ГЕНЕРАЦИЯ — Top-P (nucleus) sampling
   - Более музыкально связная генерация

Параметры по умолчанию (для 8 ГБ GPU, batch_size=4):
  d_model             = 256
  nhead               = 8
  num_encoder_layers  = 1
  num_decoder_layers  = 6
  dim_feedforward     = 1024
"""

#from __future__ import annotations

import math
import torch
import torch.nn as nn

from tokenizer import VOCAB_SIZE, PAD_TOKEN


# ══════════════════════════════════════════════════════════════
#  1. Лёгкий кастомный CNN-энкодер
# ══════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv2d → BN → GELU → optional depthwise"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, depthwise: bool = False):
        super().__init__()
        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpectrogramEncoder(nn.Module):
    """
    Лёгкий CNN → Transformer Encoder → memory

    Вход:  (B, N, F, T)
    Выход: (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_encoder_layers: int = 1,
        nhead: int = 8,
        dropout: float = 0.1,
        pretrained: bool = False,   # оставлен для совместимости API
    ):
        super().__init__()
        self.d_model = d_model

        # ── Лёгкий CNN ──────────────────────────────────────
        # (1, F, T) → постепенно уменьшаем пространственные размеры
        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  stride=2),            # F/2, T/2
            ConvBlock(32,  64,  stride=2),            # F/4, T/4
            ConvBlock(64,  128, stride=2),            # F/8, T/8
            ConvBlock(128, 256, stride=2, depthwise=True),  # F/16, T/16
            nn.AdaptiveAvgPool2d((1, 1)),             # → (B*N, 256, 1, 1)
            nn.Flatten(),                             # → (B*N, 256)
        )
        cnn_out_dim = 256

        # Проекция → d_model
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # ── Positional Encoding для сегментов ────────────────
        self.seg_pos_enc = PositionalEncoding(d_model, max_len=2048, dropout=dropout)

        # ── Transformer Encoder (лёгкий: 1 слой) ─────────────
        if num_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,   # уменьшено для экономии памяти
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=num_encoder_layers,
                norm=nn.LayerNorm(d_model),
            )
        else:
            self.transformer_encoder = None

    def forward(self, x: torch.Tensor, cnn_chunk: int = 16) -> torch.Tensor:
        """
        cnn_chunk: сколько сегментов обрабатывать через CNN за раз.
        Уменьшите до 8 или 4 если снова OOM.
        """
        B, N, F, T = x.shape
        # Обрабатываем CNN чанками — экономия пиковой памяти
        chunks = []
        for i in range(0, B * N, cnn_chunk):
            chunk = x.view(B * N, 1, F, T)[i : i + cnn_chunk]
            chunks.append(self.proj(self.cnn(chunk)))
        x = torch.cat(chunks, dim=0)   # (B*N, d_model)
        x = x.view(B, N, self.d_model) # (B, N, d_model)
        x = self.seg_pos_enc(x)
        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x)
        return x


# ══════════════════════════════════════════════════════════════
#  2. Positional Encoding
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование.
    Автоматически расширяется если seq_len > текущий max_len.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout  = nn.Dropout(p=dropout)
        self.d_model  = d_model
        self._build(max_len)

    def _build(self, max_len: int):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Если последовательность длиннее — перестраиваем PE на лету
        if seq_len > self.pe.size(1):
            self._build(seq_len + 64)   # +64 запас чтобы не перестраивать каждый раз
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════
#  3. Transformer Decoder (Pre-LN + Weight Tying)
# ══════════════════════════════════════════════════════════════
class MusicTransformerDecoder(nn.Module):
    """
    Улучшения по сравнению с v1:
    - Pre-LN (norm_first=True): стабильнее и быстрее сходится
    - Weight Tying: embedding и output_proj делят одну матрицу весов
    - Увеличен d_model и число слоёв
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding токенов
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Pre-LN Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # ← Pre-LN (ключевое улучшение)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Выходной линейный слой
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: output_proj.weight = token_embedding.weight
        # Это стандартная техника из "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)
        self.output_proj.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # output_proj.weight уже привязан к embedding — не инициализируем отдельно
        for name, p in self.transformer_decoder.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_len = tgt.size(1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device
        )

        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

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
    ScoreGenerationModel v2

    Совместима с v1 по интерфейсу (forward / generate).
    Новые параметры:
        num_encoder_layers : int = 2   — слои Transformer Encoder поверх CNN
        pretrained_encoder : bool      — использовать веса ImageNet для EfficientNet
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pretrained_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = SpectrogramEncoder(
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            dropout=dropout,
            pretrained=pretrained_encoder,
        )
        self.decoder = MusicTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        total = self.count_parameters()
        print(f"ScoreGenerationModel v2 | params: {total:,}")

    # ── Прямой проход ────────────────────────────────────────
    def forward(
        self,
        spectrograms: torch.Tensor,
        tgt: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        cnn_chunk: int = 8,
    ) -> torch.Tensor:
        memory = self.encoder(spectrograms, cnn_chunk=cnn_chunk)
        logits = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits

    # ── Авторегрессивная генерация с Top-P sampling ───────────
    @torch.no_grad()
    def generate(
        self,
        spectrograms: torch.Tensor,
        max_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.92,      # ← nucleus sampling (NEW, рекомендуется)
    ) -> torch.Tensor:
        """
        Генерирует токены авторегрессивно.

        top_p=0.92  — nucleus sampling (рекомендуется, более музыкально)
        top_k=50    — top-k sampling (как в v1, можно использовать вместо top_p)
        Можно использовать оба одновременно.
        """
        from tokenizer import BOS_TOKEN, EOS_TOKEN

        self.eval()
        device = spectrograms.device
        memory = self.encoder(spectrograms)

        generated = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decoder(tgt=generated, memory=memory)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-K фильтрация
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits = next_logits.masked_fill(
                    next_logits < values[:, -1].unsqueeze(1), float("-inf")
                )

            # Top-P (Nucleus) фильтрация
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Убираем токены после накопления top_p вероятности
                remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                # Возвращаем в исходный порядок
                next_logits = torch.zeros_like(next_logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == EOS_TOKEN:
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════
#  Конфигурации под разные GPU
# ══════════════════════════════════════════════════════════════
def build_model(preset: str = "medium") -> ScoreGenerationModel:
    """
    Готовые конфигурации:

    "small"  — 4–6 ГБ GPU, быстрое обучение
    "medium" — 8 ГБ GPU (рекомендуется, баланс качество/скорость)
    "large"  — 16+ ГБ GPU, максимальное качество
    """
    presets = {
        "small": dict(
            d_model=128, nhead=4, num_encoder_layers=1,
            num_decoder_layers=4, dim_feedforward=512, dropout=0.1,
        ),
        "medium": dict(
            d_model=256, nhead=8, num_encoder_layers=2,
            num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
        ),
        "large": dict(
            d_model=512, nhead=8, num_encoder_layers=3,
            num_decoder_layers=8, dim_feedforward=2048, dropout=0.1,
        ),
    }
    cfg = presets[preset]
    print(f"Конфигурация: {preset}")
    return ScoreGenerationModel(**cfg)