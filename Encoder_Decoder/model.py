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
ScoreGenerationModel  v3
========================

Ключевые изменения по сравнению с v2:

1. ЭНКОДЕР — предобученный ResNet18 (по умолчанию pretrained=True)
   - Принимает 3-канальный вход (спектрограмма реплицируется × 3 в dataset.py)
   - Backbone инициализируется весами ImageNet — даёт лучшие начальные признаки
   - Encoder LR должен быть меньше LR декодера (см. train.py)

2. ЭНКОДЕР — кастомный лёгкий CNN (pretrained=False)
   - Оставлен как опция когда GPU мало или данных много
   - Принимает 1-канальный вход

3. ДЕКОДЕР — без изменений (Pre-LN, Weight Tying, nucleus sampling)

4. НОРМИРОВКА — ответственность перенесена в dataset.py:
   - [0,1] (dB) + ImageNet mean/std если pretrained=True
   - [0,1] (dB) напрямую если pretrained=False

Параметры для 8 ГБ GPU (по умолчанию):
  d_model=256, nhead=8, 1 encoder слой, 6 decoder слоёв, FFN=1024
  pretrained=True  → использует ImageNet backbone
"""

import math
import torch
import torch.nn as nn

from tokenizer import VOCAB_SIZE, PAD_TOKEN


# ══════════════════════════════════════════════════════════════
#  1. CNN-энкодер спектрограммы
# ══════════════════════════════════════════════════════════════
class SpectrogramEncoder(nn.Module):
    """
    Вход:  (B, 1, F, T)  — одноканальная Mel-спектрограмма
    Выход: (B, S, d_model) — последовательность из S=28 векторов (memory декодера)

    Для входа (B, 1, 128, 216):
      ResNet18 layer4 → (B, 512, 4, 7)
      reshape         → (B, 28, 512)
      proj            → (B, 28, d_model)
    """

    def __init__(
            self,
            d_model: int = 256,
            dropout: float = 0.1,
            pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        if pretrained:
            # SWSL ResNet18 — обучен Facebook на 940М Instagram-изображениях
            # (semi-weakly supervised), даёт лучшие признаки чем стандартный ImageNet
            backbone = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models",
                "swsl_resnet18",
                verbose=False,
            )

            # Адаптируем первый слой: 3 канала → 1 канал (усредняем веса)
            orig_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv

            # Берём только feature extractor, БЕЗ avgpool и fc
            self.cnn = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                # НЕТ avgpool — сохраняем пространственную карту (B, 512, 4, 7)
            )
            cnn_out = 512

        else:
            # Лёгкий кастомный CNN без avgpool
            self.cnn = nn.Sequential(
                _ConvBlock(1, 32, stride=2),
                _ConvBlock(32, 64, stride=2),
                _ConvBlock(64, 128, stride=2),
                _ConvBlock(128, 256, stride=2),
                # НЕТ AdaptiveAvgPool — сохраняем карту (B, 256, H', W')
            )
            cnn_out = 256

        # Проекция каналов → d_model (применяется к каждой позиции)
        self.proj = nn.Sequential(
            nn.Linear(cnn_out, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Positional encoding для позиций карты признаков
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, F, T)
        (B, S, d_model)  где S = H' × W' = 4×7 = 28 для входа (1, 128, 216)
        """
        feat = self.cnn(x)  # (B, C, H', W')
        B, C, H, W = feat.shape
        # Переставляем в (B, H'*W', C) — каждый пиксель карты = один токен памяти
        feat = feat.permute(0, 2, 3, 1)  # (B, H', W', C)
        feat = feat.reshape(B, H * W, C)  # (B, S, C)
        feat = self.proj(feat)  # (B, S, d_model)
        feat = self.pos_enc(feat)  # + positional encoding
        return feat  # (B, 28, d_model)


# ──────────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ══════════════════════════════════════════════════════════════
#  2. Positional Encoding
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self._build(max_len)

    def _build(self, max_len: int):
        pe = torch.zeros(max_len, self.d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self._build(x.size(1) + 64)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ══════════════════════════════════════════════════════════════
#  3. Transformer Decoder
# ══════════════════════════════════════════════════════════════
class MusicTransformerDecoder(nn.Module):
    def __init__(
            self,
            vocab_size: int = VOCAB_SIZE,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 6,
            dim_feedforward: int = 1024,
            max_seq_len: int = 256,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for name, p in self.transformer_decoder.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        tgt_len = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device, dtype=torch.bool
        )
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer_decoder(
            tgt=x, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════
#  4. Полная модель
# ══════════════════════════════════════════════════════════════
class ScoreGenerationModel(nn.Module):
    """
    ScoreGenerationModel v5.

    Энкодер: (B, 1, F, T) - ResNet18 - (B, 28, d_model)
    Декодер: cross-attention по 28 позициям памяти (реальная информация о спектрограмме)

    """

    def __init__(
            self,
            d_model: int = 256,
            nhead: int = 8,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 1024,
            vocab_size: int = VOCAB_SIZE,
            max_seq_len: int = 256,
            dropout: float = 0.1,
            pretrained_encoder: bool = True,
            # совместимость с train.py
            num_encoder_layers: int = 0,
            cnn_chunk: int = 0,
    ):
        super().__init__()
        self.encoder = SpectrogramEncoder(
            d_model=d_model,
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
        enc_p = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_p = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        mode = "SWSL ResNet18 (1-ch, seq)" if pretrained_encoder else "custom CNN (seq)"
        print(f"ScoreGenerationModel v5 | {mode}")
        print(f"  Encoder: {enc_p:,}  |  Decoder: {dec_p:,}  |  Total: {enc_p + dec_p:,}")
        print(f"  Memory sequence: 28 позиций (было 1) → cross-attention работает")

    def forward(
            self,
            spectrograms: torch.Tensor,  # (B, 1, F, T)
            tgt: torch.Tensor,  # (B, tgt_len)
            tgt_key_padding_mask=None,
            cnn_chunk: int = 0,
    ) -> torch.Tensor:
        memory = self.encoder(spectrograms)  # (B, 28, d_model)
        return self.decoder(tgt=tgt, memory=memory,
                            tgt_key_padding_mask=tgt_key_padding_mask)

    @torch.no_grad()
    def generate(
            self,
            spectrograms: torch.Tensor,
            max_len: int = 256,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 0.92,
    ) -> torch.Tensor:
        from tokenizer import BOS_TOKEN, EOS_TOKEN
        self.eval()
        device = spectrograms.device
        memory = self.encoder(spectrograms)
        gen = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decoder(tgt=gen, memory=memory)[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
                sl = sl.masked_fill(
                    cum - torch.softmax(sl, dim=-1) > top_p, float("-inf"))
                logits = torch.zeros_like(logits).scatter_(1, si, sl)
            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            gen = torch.cat([gen, next_tok], dim=1)
            if next_tok.item() == EOS_TOKEN:
                break
        return gen

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> list:
        return [
            {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            {"params": list(self.decoder.parameters()), "lr": decoder_lr},
        ]
