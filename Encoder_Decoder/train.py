"""
train.py v5 — CNN14 + note F1
==============================
Изменения по сравнению с v4:

1. ЭНКОДЕР CNN14 (вместо ResNet18)
   - Параметр pretrained_encoder=True использует веса PANNs (AudioSet)
   - Веса автоматически скачиваются в --weights_dir при первом запуске
   - dense_time=False (по умолчанию) — 27 позиций, совместимо с весами PANNs

2. NOTE F1 МЕТРИКИ
   - pitch_f1: F1 по multiset питчей NOTE_ON (без учёта времени)
   - onset_f1: F1 по (pitch, onset_time) с допуском ±50 мс
   - Считается на validation каждую эпоху (greedy на argmax — без sampling)
   - Дополнительно — на train (с малым sample_rate, чтобы не тормозить)
   - Логируется в epoch_metrics.csv

3. РАЗДЕЛЬНЫЙ LR-CONFIG ДЛЯ ENCODER/DECODER
   - lr_encoder = 5e-5 (PANNs веса лучше дрожать аккуратно)
   - lr_decoder = 3e-4 (декодер учится с нуля)
   - Это менее агрессивно чем v4 для декодера и менее консервативно
     для энкодера, чем 1e-5 (v4) — pretrained CNN14 надёжнее, чем v4 ожидала.

Запуск:
    python train.py --dataset_dir dataset --output_dir checkpoints

    # С отключением pretrained:
    python train.py --pretrained_encoder false

    # С кастомной директорией для весов PANNs:
    python train.py --weights_dir /path/to/pretrained
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import ScoreGenerationModel
from dataset import MidiSpectrogramDataset
from tokenizer import PAD_TOKEN, VOCAB_SIZE
from metrics import NoteF1Stats


# ══════════════════════════════════════════════════════════════
#  Group-aware train/val split (по MIDI-файлам, а не сегментам)
# ══════════════════════════════════════════════════════════════
def split_by_file(dataset: MidiSpectrogramDataset, val_ratio: float, seed: int = 42):
    """
    Делит датасет на train/val ПО MIDI-ФАЙЛАМ:
        все сегменты одного файла попадают либо в train, либо в val.

    Это устраняет data leakage из-за random_split по сегментам, при котором
    соседние секунды одного произведения оказывались в разных split'ах.

    Args:
        dataset: MidiSpectrogramDataset с атрибутом .samples (список Path)
        val_ratio: доля файлов в val
        seed: для воспроизводимости

    Returns:
        (train_indices, val_indices, n_train_files, n_val_files)
    """
    import re
    import random
    from collections import defaultdict

    # Группируем индексы сегментов по file_id
    file_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample_path in enumerate(dataset.samples):
        m = re.match(r"sample_(\d+)_(\d+)", sample_path.name)
        if m:
            file_id = m.group(1)
        else:
            # Fallback: имя без расширения
            file_id = sample_path.name
        file_to_indices[file_id].append(idx)

    # Перемешиваем файлы и делим
    file_ids = sorted(file_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(file_ids)

    n_files = len(file_ids)
    n_val_files = max(1, int(n_files * val_ratio))
    val_file_ids = set(file_ids[:n_val_files])
    train_file_ids = set(file_ids[n_val_files:])

    train_indices = [i for fid in train_file_ids for i in file_to_indices[fid]]
    val_indices = [i for fid in val_file_ids for i in file_to_indices[fid]]

    return train_indices, val_indices, len(train_file_ids), len(val_file_ids)


# ══════════════════════════════════════════════════════════════
#  Конфигурация
# ══════════════════════════════════════════════════════════════
CONFIG = dict(
    # Данные
    dataset_dir="dataset",
    output_dir="checkpoints",
    weights_dir="./pretrained",  # директория для весов PANNs
    max_seq_len=256,
    max_freq_bins=128,
    max_time_steps=216,
    max_samples=0,

    # Архитектура (CNN14) — уменьшена для борьбы с overfitting
    d_model=256,
    nhead=8,
    num_decoder_layers=4,        # было 6 → меньше параметров, меньше overfitting
    dim_feedforward=768,         # было 1024
    dropout=0.3,                 # было 0.1 → главный рычаг регуляризации для transformer
    pretrained_encoder=True,
    dense_time=False,

    # Обучение
    batch_size=8,
    accum_steps=2,
    num_epochs=30,

    # ── LR (откалиброван под CNN14 + PANNs pretrained) ─────────
    lr_decoder=3e-4,
    lr_encoder=5e-5,

    weight_decay=5e-2,           # было 1e-2 → сильнее регуляризация
    warmup_steps=400,
    grad_clip=1.0,
    val_ratio=0.1,
    log_every=50,
    save_every=5,
    use_ema=True,
    ema_decay=0.999,
    reset_optimizer=False,

    # Планировщик
    scheduler_type="exponential",
    exp_gamma=0.95,

    # Фильтрация пустых сегментов
    skip_silent=False,
    silence_threshold=0.01,

    # Note F1
    onset_tolerance_ms=50,
    train_f1_every=200,          # считать F1 на train каждые N step (0 = не считать)

    # SpecAugment (аугментация спектрограмм для борьбы с overfitting)
    specaug_enabled=True,        # включить случайное маскирование time/freq полос
    specaug_n_time_masks=2,      # сколько time-полос маскировать
    specaug_time_mask_param=20,  # макс. ширина time-маски в фреймах (~20*23=460мс)
    specaug_n_freq_masks=2,      # сколько freq-полос маскировать
    specaug_freq_mask_param=15,  # макс. ширина freq-маски в mel-bins

    # Mixed precision / отладка
    use_amp=True,                # False = отключить mixed precision (для отладки NaN)
    amp_dtype="bfloat16",        # "bfloat16" (рекомендовано для CNN14) или "float16"
    debug_first_batch=True,      # печатать статистику активаций на первом батче

    # Split (anti-leakage)
    split_seed=42,               # seed для group-split по MIDI-файлам

    # Early stopping
    patience=8,                  # эпох без улучшения val_loss до остановки (0 = выкл.)
)


# ══════════════════════════════════════════════════════════════
#  Планировщики
# ══════════════════════════════════════════════════════════════
class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))

        super().__init__(optimizer, lr_lambda)


class WarmupExponentialScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, gamma: float, steps_per_epoch: int):
        self._warmup = warmup_steps
        self._gamma = gamma
        self._spe = max(1, steps_per_epoch)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            elapsed = step - warmup_steps
            epoch_frac = elapsed / self._spe
            return self._gamma ** epoch_frac

        super().__init__(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def get_model(self):
        return self.shadow


# ══════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════
def accuracy_no_pad(logits, targets):
    preds = logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN
    correct = (preds[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


class MetricsLogger:
    """Логгер с поддержкой F1-метрик (расширен по сравнению с v4)."""

    def __init__(self, path: Path):
        self.path = path
        self._step_path = path.parent / "step_metrics.csv"
        self.global_step = 0
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "train_pitch_f1", "train_onset_f1",
                "val_pitch_f1", "val_onset_f1",
                "lr_dec", "lr_enc", "elapsed_s", "nan_batches",
            ])
        with open(self._step_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "global_step", "epoch", "step", "loss", "acc",
                "pitch_f1", "onset_f1", "lr_dec",
            ])

    def log_step(self, epoch, step, loss, acc, lr_dec,
                 pitch_f1=None, onset_f1=None):
        self.global_step += 1
        with open(self._step_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.global_step, epoch, step,
                f"{loss:.6f}", f"{acc:.6f}",
                f"{pitch_f1:.6f}" if pitch_f1 is not None else "",
                f"{onset_f1:.6f}" if onset_f1 is not None else "",
                f"{lr_dec:.8f}",
            ])

    def log_epoch(self, epoch, tl, ta, vl, va,
                  train_pf1, train_of1, val_pf1, val_of1,
                  lr_dec, lr_enc, elapsed, nan_batches=0):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tl:.6f}", f"{ta:.6f}", f"{vl:.6f}", f"{va:.6f}",
                f"{train_pf1:.6f}", f"{train_of1:.6f}",
                f"{val_pf1:.6f}", f"{val_of1:.6f}",
                f"{lr_dec:.8f}", f"{lr_enc:.8f}", f"{elapsed:.1f}",
                nan_batches,
            ])


# ══════════════════════════════════════════════════════════════
#  Мониторинг градиентов (без изменений)
# ══════════════════════════════════════════════════════════════
class GradMonitor:
    def __init__(self, path: Path):
        self.path = path
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow([
                "global_step",
                "enc_grad_min", "enc_grad_max", "enc_grad_mean",
                "dec_grad_min", "dec_grad_max", "dec_grad_mean",
            ])
        self.global_step = 0

    def log(self, model):
        self.global_step += 1
        enc_grads, dec_grads = [], []
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.data.abs()
                if name.startswith("encoder"):
                    enc_grads.append(g)
                else:
                    dec_grads.append(g)

        def stats(grads):
            if not grads:
                return 0.0, 0.0, 0.0
            all_g = torch.cat([g.flatten() for g in grads])
            return all_g.min().item(), all_g.max().item(), all_g.mean().item()

        e_min, e_max, e_mean = stats(enc_grads)
        d_min, d_max, d_mean = stats(dec_grads)

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.global_step,
                f"{e_min:.6f}", f"{e_max:.6f}", f"{e_mean:.6f}",
                f"{d_min:.6f}", f"{d_max:.6f}", f"{d_mean:.6f}",
            ])

        if d_max > 10.0:
            print(f"  [!] ГРАДИЕНТ АНОМАЛИЯ: dec_grad_max={d_max:.2f}")
        if e_max > 10.0:
            print(f"  [!] ГРАДИЕНТ АНОМАЛИЯ: enc_grad_max={e_max:.2f}")


# ══════════════════════════════════════════════════════════════
#  SpecAugment (Park et al., 2019) — аугментация спектрограмм
# ══════════════════════════════════════════════════════════════
def spec_augment(
        spec: torch.Tensor,
        n_time_masks: int = 2,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        freq_mask_param: int = 15,
) -> torch.Tensor:
    """
    Применяет SpecAugment к батчу спектрограмм.
    Маскирует случайные time/freq полосы → модель не может полагаться
    на конкретные узкие частотные диапазоны или временные позиции.

    Args:
        spec: (B, 1, F, T) — мел-спектрограмма
        n_time_masks: количество time-полос для маскирования
        time_mask_param: максимальная ширина time-маски (фреймов)
        n_freq_masks: количество freq-полос
        freq_mask_param: максимальная ширина freq-маски (mel-bins)

    Returns:
        spec с маскированными полосами (заменены на 0 = тишина в нашей нормировке)
    """
    if spec.dim() != 4:
        return spec
    B, C, F, T = spec.shape

    # Time masks
    for _ in range(n_time_masks):
        t = torch.randint(0, time_mask_param + 1, (B,), device=spec.device)
        t0 = (torch.rand(B, device=spec.device) * (T - t.float()).clamp(min=1)).long()
        for b in range(B):
            if t[b] > 0:
                spec[b, :, :, t0[b]:t0[b] + t[b]] = 0.0

    # Frequency masks
    for _ in range(n_freq_masks):
        f = torch.randint(0, freq_mask_param + 1, (B,), device=spec.device)
        f0 = (torch.rand(B, device=spec.device) * (F - f.float()).clamp(min=1)).long()
        for b in range(B):
            if f[b] > 0:
                spec[b, :, f0[b]:f0[b] + f[b], :] = 0.0

    return spec


# ══════════════════════════════════════════════════════════════
#  Шаг обучения
# ══════════════════════════════════════════════════════════════
def train_step(model, batch, criterion, scaler, device, accum_steps, step_idx,
               return_preds: bool = False, use_amp: bool = True,
               amp_dtype=None, debug: bool = False,
               specaug_cfg: dict = None):
    """
    Возвращает (loss, acc, do_update, preds_or_None).
    use_amp=False → отключает mixed precision (полезно для отладки NaN).
    amp_dtype: torch.float16 или torch.bfloat16.
    specaug_cfg: dict с параметрами SpecAugment, или None (без аугментации).
    debug=True → печатает статистику активаций каждого слоя энкодера.
    """
    specs, src, tgt, pad_mask = [t.to(device) for t in batch]

    # ── Применяем SpecAugment (только в train mode, model сам train()) ────
    if specaug_cfg is not None and model.training:
        # Клонируем, чтобы не модифицировать данные в DataLoader
        specs = specs.clone()
        specs = spec_augment(
            specs,
            n_time_masks=specaug_cfg.get("n_time_masks", 2),
            time_mask_param=specaug_cfg.get("time_mask_param", 20),
            n_freq_masks=specaug_cfg.get("n_freq_masks", 2),
            freq_mask_param=specaug_cfg.get("freq_mask_param", 15),
        )

    if debug:
        with torch.no_grad():
            print(f"  [DBG] specs: shape={tuple(specs.shape)} "
                  f"min={specs.min().item():.4f} max={specs.max().item():.4f} "
                  f"mean={specs.mean().item():.4f} "
                  f"nan={torch.isnan(specs).any().item()}")
            print(f"  [DBG] tgt: shape={tuple(tgt.shape)} "
                  f"min={tgt.min().item()} max={tgt.max().item()} "
                  f"non_pad={(tgt != 0).sum().item()}/{tgt.numel()}")

    autocast_enabled = use_amp and (device.type == "cuda")
    autocast_kwargs = {"device_type": device.type, "enabled": autocast_enabled}
    if amp_dtype is not None and autocast_enabled:
        autocast_kwargs["dtype"] = amp_dtype

    with torch.autocast(**autocast_kwargs):
        logits = model(specs, src, tgt_key_padding_mask=pad_mask, debug=debug)
        if debug:
            with torch.no_grad():
                print(f"  [DBG] logits: shape={tuple(logits.shape)} "
                      f"min={logits.min().item():.4f} "
                      f"max={logits.max().item():.4f} "
                      f"nan={torch.isnan(logits).any().item()} "
                      f"inf={torch.isinf(logits).any().item()}")
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)) / accum_steps

    if debug:
        print(f"  [DBG] loss: {loss.item() * accum_steps:.6f}")

    if not math.isfinite(loss.item() * accum_steps):
        return float("nan"), 0.0, (step_idx + 1) % accum_steps == 0, None

    # GradScaler.scale() is no-op if scaler is disabled (bf16 / FP32),
    # so we can call it unconditionally — it's safe.
    scaler.scale(loss).backward()
    acc = accuracy_no_pad(logits.detach(), tgt)
    preds = logits.detach().argmax(-1) if return_preds else None
    return loss.item() * accum_steps, acc, (step_idx + 1) % accum_steps == 0, preds


# ══════════════════════════════════════════════════════════════
#  Валидация с F1
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion, device, onset_tolerance_sec: float):
    """
    Возвращает: (val_loss, val_acc, pitch_f1, onset_f1)
    """
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    f1_stats = NoteF1Stats(onset_tolerance_sec=onset_tolerance_sec)

    for batch in loader:
        specs, src, tgt, pad_mask = [t.to(device) for t in batch]
        logits = model(specs, src, tgt_key_padding_mask=pad_mask)
        loss_val = criterion(
            logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)).item()
        if math.isfinite(loss_val):
            tot_loss += loss_val
            tot_acc += accuracy_no_pad(logits, tgt)
            n += 1
        # F1 по argmax-предсказаниям (teacher-forced)
        preds = logits.argmax(-1)
        f1_stats.update(preds, tgt)

    model.train()
    summary = f1_stats.summary()
    if n > 0:
        return tot_loss / n, tot_acc / n, summary["pitch_f1"], summary["onset_f1"]
    return 0.0, 0.0, summary["pitch_f1"], summary["onset_f1"]


# ══════════════════════════════════════════════════════════════
#  Главный цикл
# ══════════════════════════════════════════════════════════════
def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    torch.cuda.empty_cache()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir / "epoch_metrics.csv")
    grad_mon = GradMonitor(out_dir / "grad_monitor.csv")

    onset_tol_sec = cfg["onset_tolerance_ms"] / 1000.0

    # ── Данные ───────────────────────────────────────────────
    full_dataset = MidiSpectrogramDataset(
        dataset_root=cfg["dataset_dir"],
        max_seq_len=cfg["max_seq_len"],
        max_freq_bins=cfg["max_freq_bins"],
        max_time_steps=cfg["max_time_steps"],
        skip_silent=cfg.get("skip_silent", False),
        silence_threshold=cfg.get("silence_threshold", 0.01),
    )

    # ── Group-split по MIDI-файлам (anti-leakage) ────────────
    # Все сегменты одного файла попадают целиком в train ИЛИ в val.
    # Без этого сегменты одной пьесы попадают и туда и туда → лик контекста.
    split_seed = cfg.get("split_seed", 42)
    train_indices, val_indices, n_train_files, n_val_files = split_by_file(
        full_dataset, val_ratio=cfg["val_ratio"], seed=split_seed,
    )
    print(f"Split по файлам (seed={split_seed}): "
          f"{n_train_files} train-файлов, {n_val_files} val-файлов")

    # Применяем max_samples ПОСЛЕ split, чтобы val оставался корректным
    max_samples = cfg.get("max_samples", 0)
    if max_samples > 0:
        n_val_keep = max(1, int(max_samples * cfg["val_ratio"]))
        n_train_keep = max_samples - n_val_keep
        if n_train_keep < len(train_indices):
            train_indices = train_indices[:n_train_keep]
        if n_val_keep < len(val_indices):
            val_indices = val_indices[:n_val_keep]
        print(f"max_samples: train усечён до {len(train_indices)}, "
              f"val до {len(val_indices)}")

    from torch.utils.data import Subset
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    n_train = len(train_ds)
    n_val = len(val_ds)

    _nw = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=_nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=_nw, pin_memory=True)
    print(f"Train: {n_train}  |  Val: {n_val}  |  "
          f"Eff.batch: {cfg['batch_size'] * cfg['accum_steps']}")

    # ── Модель ───────────────────────────────────────────────
    model = ScoreGenerationModel(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        vocab_size=VOCAB_SIZE,
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        pretrained_encoder=cfg["pretrained_encoder"],
        weights_dir=cfg["weights_dir"],
        dense_time=cfg["dense_time"],
    ).to(device)

    ema = ModelEMA(model, decay=cfg["ema_decay"]) if cfg["use_ema"] else None

    optimizer = torch.optim.AdamW(
        model.get_param_groups(cfg["lr_encoder"], cfg["lr_decoder"]),
        weight_decay=cfg["weight_decay"],
    )

    steps_per_epoch = len(train_loader) // cfg["accum_steps"]
    total_steps = steps_per_epoch * cfg["num_epochs"]

    if cfg.get("scheduler_type", "exponential") == "exponential":
        scheduler = WarmupExponentialScheduler(
            optimizer,
            warmup_steps=cfg["warmup_steps"],
            gamma=cfg.get("exp_gamma", 0.95),
            steps_per_epoch=steps_per_epoch,
        )
        sched_name = f"WarmupExponential(gamma={cfg.get('exp_gamma', 0.95)})"
    else:
        scheduler = WarmupCosineScheduler(
            optimizer, cfg["warmup_steps"], total_steps)
        sched_name = "WarmupCosine"

    use_amp = cfg.get("use_amp", True)
    amp_dtype_str = cfg.get("amp_dtype", "bfloat16").lower()
    if amp_dtype_str == "bfloat16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_str == "float16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float16', got '{amp_dtype_str}'")

    # GradScaler нужен только для float16 (bfloat16 не имеет проблем с underflow градиентов).
    scaler_enabled = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)

    print(f"LR: decoder={cfg['lr_decoder']:.0e}  encoder={cfg['lr_encoder']:.0e}")
    print(f"Warmup: {cfg['warmup_steps']} steps  |  Scheduler: {sched_name}")
    print(f"Grad clip: {cfg['grad_clip']}  |  Label smoothing: 0.1")
    print(f"Note F1: pitch (multiset) + onset (±{cfg['onset_tolerance_ms']}мс)")
    if use_amp:
        print(f"AMP: ON ({amp_dtype_str}, GradScaler={'ON' if scaler_enabled else 'OFF'})")
    else:
        print(f"AMP: OFF (FP32)")

    # ── Загрузка чекпоинта ───────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    ckpt_path = out_dir / "last.pt"
    best_path = out_dir / "best_model.pt"
    best_f1_path = out_dir / "best_f1_model.pt"
    reset_optimizer = cfg.get("reset_optimizer", False)

    if ckpt_path.exists() and not reset_optimizer:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        if ema and "ema" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema"])
        print(f"Возобновление с эпохи {start_epoch} "
              f"(best_loss={best_val_loss:.4f}, best_f1={best_val_f1:.4f})")
    elif best_path.exists() and reset_optimizer:
        weights = torch.load(best_path, map_location=device)
        model.load_state_dict(weights)
        if ema:
            ema.shadow.load_state_dict(weights)
        start_epoch = 0
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        print(f"Загружены лучшие веса из best_model.pt; оптимизатор СБРОШЕН")
    else:
        print("Чекпоинт не найден — обучение с нуля")

    # ── Цикл ─────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    train_f1_every = cfg.get("train_f1_every", 0)
    debug_first_batch = cfg.get("debug_first_batch", True)

    # Early stopping
    patience = cfg.get("patience", 0)
    epochs_without_improvement = 0
    if patience > 0:
        print(f"Early stopping: patience={patience} эпох без улучшения val_loss")

    # Конфиг SpecAugment
    specaug_cfg = None
    if cfg.get("specaug_enabled", False):
        specaug_cfg = {
            "n_time_masks": cfg.get("specaug_n_time_masks", 2),
            "time_mask_param": cfg.get("specaug_time_mask_param", 20),
            "n_freq_masks": cfg.get("specaug_n_freq_masks", 2),
            "freq_mask_param": cfg.get("specaug_freq_mask_param", 15),
        }
        print(f"SpecAugment: time={specaug_cfg['n_time_masks']}×"
              f"≤{specaug_cfg['time_mask_param']}, freq="
              f"{specaug_cfg['n_freq_masks']}×≤{specaug_cfg['freq_mask_param']}")
    else:
        print(f"SpecAugment: OFF")

    for epoch in range(start_epoch, cfg["num_epochs"]):
        ep_loss, ep_acc = 0.0, 0.0
        ep_valid_steps = 0
        ep_nan_count = 0
        # Аккумулятор F1 на train (по подвыборке батчей)
        train_f1_stats = NoteF1Stats(onset_tolerance_sec=onset_tol_sec)
        last_pf1, last_of1 = 0.0, 0.0
        t0 = time.time()

        for step_idx, batch in enumerate(train_loader):
            # Считаем preds на каждом N-ом шаге для F1
            need_preds = (train_f1_every > 0 and (step_idx + 1) % train_f1_every == 0)
            # Debug на первом батче самой первой эпохи (или после reset)
            do_debug = debug_first_batch and epoch == start_epoch and step_idx == 0
            if do_debug:
                print("\n  ━━━━━ DEBUG: первый forward (диагностика NaN) ━━━━━")
            loss, acc, do_update, preds = train_step(
                model, batch, criterion, scaler, device,
                cfg["accum_steps"], step_idx,
                return_preds=need_preds,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                debug=do_debug,
                specaug_cfg=specaug_cfg,
            )
            if do_debug:
                print("  ━━━━━ end DEBUG ━━━━━\n")

            if math.isfinite(loss):
                ep_loss += loss
                ep_acc += acc
                ep_valid_steps += 1
            else:
                ep_nan_count += 1
                optimizer.zero_grad()
                continue

            # Обновляем F1 на train, если посчитали preds
            if need_preds and preds is not None:
                tgt = batch[2].to(device)
                train_f1_stats.update(preds, tgt)
                summary = train_f1_stats.summary()
                last_pf1, last_of1 = summary["pitch_f1"], summary["onset_f1"]

            if do_update:
                if scaler_enabled:
                    # FP16: unscale → clip → step → update
                    scaler.unscale_(optimizer)
                    total_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["grad_clip"])
                    if math.isfinite(total_norm.item()):
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        if ema:
                            ema.update(model)
                        if (step_idx + 1) % cfg["log_every"] == 0:
                            grad_mon.log(model)
                    else:
                        print(f"  [!] Grad norm = {total_norm.item():.2f} (inf/nan), пропуск")
                        scaler.update()
                else:
                    # FP32 / bfloat16: GradScaler не используется
                    total_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["grad_clip"])
                    if math.isfinite(total_norm.item()):
                        optimizer.step()
                        scheduler.step()
                        if ema:
                            ema.update(model)
                        if (step_idx + 1) % cfg["log_every"] == 0:
                            grad_mon.log(model)
                    else:
                        print(f"  [!] Grad norm = {total_norm.item():.2f} (inf/nan), пропуск")

                optimizer.zero_grad()

            if (step_idx + 1) % cfg["log_every"] == 0:
                lr_dec = optimizer.param_groups[1]["lr"]
                lr_enc = optimizer.param_groups[0]["lr"]
                logger.log_step(epoch + 1, step_idx + 1, loss, acc, lr_dec,
                                pitch_f1=last_pf1 if train_f1_every > 0 else None,
                                onset_f1=last_of1 if train_f1_every > 0 else None)
                f1_str = (f"  pf1={last_pf1:.3f} of1={last_of1:.3f}"
                          if train_f1_every > 0 else "")
                print(f"  Epoch {epoch + 1} step {step_idx + 1}/"
                      f"{len(train_loader)} | "
                      f"loss={loss:.4f}  acc={acc:.3f}  "
                      f"lr_dec={lr_dec:.2e}  lr_enc={lr_enc:.2e}  "
                      f"nan={ep_nan_count}{f1_str}")

        # ── Статистика эпохи ─────────────────────────────────
        n_steps = max(1, ep_valid_steps)
        avg_loss = ep_loss / n_steps
        avg_acc = ep_acc / n_steps

        train_summary = train_f1_stats.summary()
        train_pf1 = train_summary["pitch_f1"]
        train_of1 = train_summary["onset_f1"]

        val_model = ema.get_model() if ema else model
        val_loss, val_acc, val_pf1, val_of1 = validate(
            val_model, val_loader, criterion, device, onset_tol_sec)
        elapsed = time.time() - t0

        lr_dec = optimizer.param_groups[1]["lr"]
        lr_enc = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch + 1, avg_loss, avg_acc, val_loss, val_acc,
                         train_pf1, train_of1, val_pf1, val_of1,
                         lr_dec, lr_enc, elapsed, ep_nan_count)

        nan_pct = ep_nan_count / max(1, len(train_loader)) * 100
        nan_warn = f"  [!] NaN: {ep_nan_count} ({nan_pct:.1f}%)" if ep_nan_count > 0 else ""

        print(f"Epoch {epoch + 1:3d} | "
              f"train loss={avg_loss:.4f} acc={avg_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"val pitch_f1={val_pf1:.3f} onset_f1={val_of1:.3f} | "
              f"{elapsed:.1f}s{nan_warn}")

        if nan_pct > 5.0:
            print(f"  [!] ВНИМАНИЕ: {nan_pct:.1f}% батчей дали NaN!")

        if epoch > 0 and avg_loss > 5.5:
            print(f"  [!] ВОЗМОЖНЫЙ ВЗРЫВ: train loss={avg_loss:.4f} > 5.5")

        # ── Сохранение ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_m = ema.get_model() if ema else model
            torch.save(save_m.state_dict(), out_dir / "best_model.pt")
            print(f"  * Лучшая модель по loss (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0:
                print(f"  · val_loss не улучшен ({epochs_without_improvement}/{patience})")

        # Отдельно сохраняем лучшую по onset_f1 — это финальная метрика качества
        if val_of1 > best_val_f1:
            best_val_f1 = val_of1
            save_m = ema.get_model() if ema else model
            torch.save(save_m.state_dict(), out_dir / "best_f1_model.pt")
            print(f"  * Лучшая модель по F1 (val_onset_f1={val_of1:.4f})")

        if (epoch + 1) % cfg["save_every"] == 0:
            ckpt_data = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
            }
            if ema:
                ckpt_data["ema"] = ema.shadow.state_dict()
            torch.save(ckpt_data, ckpt_path)

        # ── Early stopping ──────────────────────────────────
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"\n  ━━ EARLY STOPPING ━━")
            print(f"  val_loss не улучшался {patience} эпох подряд.")
            print(f"  Лучший val_loss: {best_val_loss:.4f}, val_onset_f1: {best_val_f1:.4f}")
            print(f"  Остановка на эпохе {epoch + 1}.")
            break

    print("Обучение завершено!")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, val in CONFIG.items():
        if isinstance(val, bool):
            parser.add_argument(
                f"--{key}", type=lambda x: x.lower() != "false", default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    train(vars(args))