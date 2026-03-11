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
import torchvision.models as tv_models

from tokenizer import VOCAB_SIZE, PAD_TOKEN


# ══════════════════════════════════════════════════════════════
#  1. Энкодер спектрограмм
# ══════════════════════════════════════════════════════════════
class SpectrogramEncoder(nn.Module):
    """
    Два режима:
    pretrained=True  — ResNet18 с весами ImageNet, вход (B*N, 3, F, T)
    pretrained=False — лёгкий кастомный CNN,      вход (B*N, 1, F, T)

    Выход обоих вариантов: (B, N, d_model)
    """

    def __init__(
            self,
            d_model: int = 256,
            num_encoder_layers: int = 1,
            nhead: int = 8,
            dropout: float = 0.1,
            pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pretrained = pretrained

        if pretrained:
            # ── Предобученный ResNet18 ────────────────────────
            weights = tv_models.ResNet18_Weights.DEFAULT
            backbone = tv_models.resnet18(weights=weights)
            # Убираем финальный классификатор
            self.backbone = nn.Sequential(
                backbone.conv1,  # ожидает 3 канала
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                backbone.avgpool,  # → (B*N, 512, 1, 1)
                nn.Flatten(),  # → (B*N, 512)
            )
            cnn_out = 512
        else:
            # ── Лёгкий кастомный CNN ──────────────────────────
            self.backbone = nn.Sequential(
                _ConvBlock(1, 32, stride=2),
                _ConvBlock(32, 64, stride=2),
                _ConvBlock(64, 128, stride=2),
                _ConvBlock(128, 256, stride=2, depthwise=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            cnn_out = 256

        # Проекция → d_model
        self.proj = nn.Sequential(
            nn.Linear(cnn_out, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Positional Encoding для временны́х сегментов
        self.seg_pos_enc = PositionalEncoding(d_model, max_len=2048, dropout=dropout)

        # Лёгкий Transformer Encoder поверх CNN (1 слой — видит контекст соседних сегментов)
        if num_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer_enc = nn.TransformerEncoder(
                enc_layer, num_layers=num_encoder_layers,
                norm=nn.LayerNorm(d_model),
            )
        else:
            self.transformer_enc = None

    def forward(self, x: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        """
        x: (B, N, C, F, T)  — C=3 если pretrained, C=1 если нет
        """
        B, N, C, F, T = x.shape
        # Обрабатываем CNN чанками для экономии пиковой памяти
        x_flat = x.view(B * N, C, F, T)
        outs = []
        for i in range(0, B * N, chunk):
            outs.append(self.proj(self.backbone(x_flat[i: i + chunk])))
        x = torch.cat(outs, dim=0).view(B, N, self.d_model)  # (B, N, d_model)
        x = self.seg_pos_enc(x)
        if self.transformer_enc is not None:
            x = self.transformer_enc(x)
        return x


# ──────────────────────────────────────────────────────────────
#  Блок кастомного CNN (используется только когда pretrained=False)
# ──────────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, depthwise=False):
        super().__init__()
        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ══════════════════════════════════════════════════════════════
#  2. Positional Encoding
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self._build(max_len)

    def _build(self, max_len: int):
        pe = torch.zeros(max_len, self.d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.d_model, 2).float()
                        * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self._build(x.size(1) + 64)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ══════════════════════════════════════════════════════════════
#  3. Transformer Decoder (Pre-LN + Weight Tying)
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
            norm_first=True,  # Pre-LN: стабильнее градиенты
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight Tying — уменьшает переобучение, стандарт для LM
        self.output_proj.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for name, p in self.transformer_decoder.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_len = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer_decoder(
            tgt=x, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(x)  # (B, tgt_len, vocab_size)


# ══════════════════════════════════════════════════════════════
#  4. Полная модель
# ══════════════════════════════════════════════════════════════
class ScoreGenerationModel(nn.Module):
    """
    ScoreGenerationModel v3.

    ВАЖНО: если pretrained=True, dataset.py должен возвращать
    спектрограммы формы (N, 3, F, T) с ImageNet-нормировкой.
    Это управляется параметром imagenet_norm=True в MidiSpectrogramDataset.
    """

    def __init__(
            self,
            d_model: int = 256,
            nhead: int = 8,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 1024,
            vocab_size: int = VOCAB_SIZE,
            max_seq_len: int = 256,
            dropout: float = 0.1,
            pretrained_encoder: bool = True,  # ← по умолчанию теперь True
    ):
        super().__init__()

        self.pretrained_encoder = pretrained_encoder

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
        enc_p = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_p = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        mode = "pretrained ResNet18" if pretrained_encoder else "custom CNN"
        print(f"ScoreGenerationModel v3 | {mode}")
        print(f"  Encoder: {enc_p:,} params  |  Decoder: {dec_p:,} params  |  Total: {total:,}")

    def forward(
            self,
            spectrograms: torch.Tensor,  # (B, N, C, F, T)
            tgt: torch.Tensor,  # (B, tgt_len)
            tgt_key_padding_mask=None,
            cnn_chunk: int = 8,
    ) -> torch.Tensor:
        memory = self.encoder(spectrograms, chunk=cnn_chunk)
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
                vals, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))

            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                sorted_l = sorted_l.masked_fill(
                    cum - torch.softmax(sorted_l, dim=-1) > top_p, float("-inf"))
                logits = torch.zeros_like(logits).scatter_(1, sorted_i, sorted_l)

            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            gen = torch.cat([gen, next_tok], dim=1)
            if next_tok.item() == EOS_TOKEN:
                break

        return gen

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> list:
        """
        Возвращает param_groups для AdamW с раздельными LR.
        encoder_lr  — обычно 1e-4  (backbone дообучается медленно)
        decoder_lr  — обычно 1e-3  (трансформер учится быстро)
        """
        return [
            {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            {"params": list(self.decoder.parameters()), "lr": decoder_lr},
        ]


# ══════════════════════════════════════════════════════════════
#  Готовые конфигурации
# ══════════════════════════════════════════════════════════════
def build_model(preset: str = "medium", pretrained: bool = True) -> ScoreGenerationModel:
    """
    "small"  — 4–6 ГБ GPU
    "medium" — 8 ГБ GPU (рекомендуется)
    "large"  — 16+ ГБ GPU
    """
    presets = {
        "small": dict(d_model=128, nhead=4, num_encoder_layers=1,
                      num_decoder_layers=4, dim_feedforward=512),
        "medium": dict(d_model=256, nhead=8, num_encoder_layers=1,
                       num_decoder_layers=6, dim_feedforward=1024),
        "large": dict(d_model=512, nhead=8, num_encoder_layers=2,
                      num_decoder_layers=8, dim_feedforward=2048),
    }
    return ScoreGenerationModel(**presets[preset], pretrained_encoder=pretrained)
