#from __future__ import annotations
"""
train.py
========
Цикл обучения модели ScoreGenerationModel.
Логирует метрики в CSV и строит графики после обучения.

Запуск:
    python train.py \
        --dataset_dir /path/to/dataset \
        --output_dir  /path/to/checkpoints
"""

"""
train.py  v3
============
Ключевые изменения:
  - lr_decoder = 1e-3
  - lr_encoder = 1e-4  (backbone дообучается медленно)
  - warmup_steps = 300
  - imagenet_norm = True — датасет возвращает 3-канальный ImageNet-нормированный тензор
  - pretrained_encoder = True — используем ResNet18 с весами ImageNet
  - max_time_steps = 216 (для Mel+hop=512: 5 сек ≈ 216 фреймов, было 256/399)

ВАЖНО: после смены audio_processor на Mel нужно заново запустить prepare_dataset.py!
"""

import argparse
import copy
import csv
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


# ══════════════════════════════════════════════════════════════
#  Конфигурация
# ══════════════════════════════════════════════════════════════
CONFIG = dict(
    # Данные
    dataset_dir     = "dataset",
    output_dir      = "checkpoints",
    max_seq_len     = 256,
    max_segments    = 0,        # 0 = автоопределение
    max_freq_bins   = 128,
    max_time_steps  = 216,      # Mel + hop=512 + segment=5 сек ≈ 216 фреймов

    # Архитектура
    d_model             = 256,
    nhead               = 8,
    num_encoder_layers  = 1,
    num_decoder_layers  = 6,
    dim_feedforward     = 1024,
    dropout             = 0.1,
    pretrained_encoder  = True,   # ← использовать предобученный ResNet18
    imagenet_norm       = True,   # ← нормировка под ImageNet (нужна при pretrained=True)

    # Обучение
    batch_size      = 2,
    accum_steps     = 4,          # эффективный batch = 8
    cnn_chunk       = 8,
    num_epochs      = 60,

    # LR — КЛЮЧЕВОЕ ИЗМЕНЕНИЕ по совету научрука
    lr_decoder      = 1e-3,       # ← Transformer decoder: высокий LR
    lr_encoder      = 1e-4,       # ← ResNet18 backbone: низкий LR (дообучаем осторожно)

    weight_decay    = 1e-2,
    warmup_steps    = 300,        # было 1000 — слишком долго (~21 эпоха warmup)
    grad_clip       = 1.0,
    val_ratio       = 0.1,
    log_every       = 50,
    save_every      = 5,
    use_ema         = True,
    ema_decay       = 0.999,
)


# ══════════════════════════════════════════════════════════════
#  Планировщик: warmup + cosine decay
# ══════════════════════════════════════════════════════════════
class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        import math
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))
        super().__init__(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════
#  EMA весов
# ══════════════════════════════════════════════════════════════
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
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
    preds   = logits.argmax(dim=-1)
    mask    = targets != PAD_TOKEN
    correct = (preds[mask] == targets[mask]).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


class MetricsLogger:
    def __init__(self, path: Path):
        self.path       = path
        self._step_path = path.parent / "step_metrics.csv"
        self.global_step = 0
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","train_loss","train_acc","val_loss","val_acc","lr_dec","lr_enc","elapsed_s"])
        with open(self._step_path, "w", newline="") as f:
            csv.writer(f).writerow(["global_step","epoch","step","loss","acc","lr_dec"])

    def log_step(self, epoch, step, loss, acc, lr_dec):
        self.global_step += 1
        with open(self._step_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [self.global_step, epoch, step, f"{loss:.6f}", f"{acc:.6f}", f"{lr_dec:.8f}"])

    def log_epoch(self, epoch, tl, ta, vl, va, lr_dec, lr_enc, elapsed):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tl:.6f}", f"{ta:.6f}", f"{vl:.6f}", f"{va:.6f}",
                 f"{lr_dec:.8f}", f"{lr_enc:.8f}", f"{elapsed:.1f}"])


# ══════════════════════════════════════════════════════════════
#  Шаг обучения
# ══════════════════════════════════════════════════════════════
def train_step(model, batch, criterion, scaler, device, accum_steps, step_idx, cnn_chunk=8):
    specs, src, tgt, pad_mask = [t.to(device) for t in batch]
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(specs, src, tgt_key_padding_mask=pad_mask, cnn_chunk=cnn_chunk)
        loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)) / accum_steps
    scaler.scale(loss).backward()
    acc = accuracy_no_pad(logits.detach(), tgt)
    return loss.item() * accum_steps, acc, (step_idx + 1) % accum_steps == 0


# ══════════════════════════════════════════════════════════════
#  Валидация
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for batch in loader:
        specs, src, tgt, pad_mask = [t.to(device) for t in batch]
        logits = model(specs, src, tgt_key_padding_mask=pad_mask)
        tot_loss += criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)).item()
        tot_acc  += accuracy_no_pad(logits, tgt)
        n += 1
    model.train()
    return tot_loss / n, tot_acc / n


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

    # ── Данные ───────────────────────────────────────────────
    full_dataset = MidiSpectrogramDataset(
        dataset_root   = cfg["dataset_dir"],
        max_seq_len    = cfg["max_seq_len"],
        max_segments   = cfg["max_segments"],
        max_freq_bins  = cfg["max_freq_bins"],
        max_time_steps = cfg["max_time_steps"],
        imagenet_norm  = cfg["imagenet_norm"],
    )
    n_val   = max(1, int(len(full_dataset) * cfg["val_ratio"]))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    _nw = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=_nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=_nw, pin_memory=True)
    print(f"Train: {n_train}  |  Val: {n_val}  |  Eff.batch: {cfg['batch_size']*cfg['accum_steps']}")

    # ── Модель ───────────────────────────────────────────────
    model = ScoreGenerationModel(
        d_model            = cfg["d_model"],
        nhead              = cfg["nhead"],
        num_encoder_layers = cfg["num_encoder_layers"],
        num_decoder_layers = cfg["num_decoder_layers"],
        dim_feedforward    = cfg["dim_feedforward"],
        vocab_size         = VOCAB_SIZE,
        max_seq_len        = cfg["max_seq_len"],
        dropout            = cfg["dropout"],
        pretrained_encoder = cfg["pretrained_encoder"],
    ).to(device)

    ema = ModelEMA(model, decay=cfg["ema_decay"]) if cfg["use_ema"] else None

    # ── Раздельный LR: encoder (backbone) vs decoder ─────────
    optimizer = torch.optim.AdamW(
        model.get_param_groups(
            encoder_lr = cfg["lr_encoder"],   # 1e-4 — backbone медленно
            decoder_lr = cfg["lr_decoder"],   # 1e-3 — transformer быстро
        ),
        weight_decay = cfg["weight_decay"],
    )

    total_steps = (len(train_loader) // cfg["accum_steps"]) * cfg["num_epochs"]
    scheduler   = WarmupCosineScheduler(optimizer, cfg["warmup_steps"], total_steps)
    scaler      = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)

    print(f"LR decoder={cfg['lr_decoder']:.0e}  encoder={cfg['lr_encoder']:.0e}  "
          f"warmup={cfg['warmup_steps']} steps")

    # ── Загрузка чекпоинта ───────────────────────────────────
    start_epoch  = 0
    best_val_loss = float("inf")
    ckpt_path    = out_dir / "last.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if ema and "ema" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema"])
        print(f"Возобновление с эпохи {start_epoch} (best val_loss={best_val_loss:.4f})")

    # ── Цикл ─────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, cfg["num_epochs"]):
        ep_loss, ep_acc = 0.0, 0.0
        t0 = time.time()

        for step_idx, batch in enumerate(train_loader):
            loss, acc, do_update = train_step(
                model, batch, criterion, scaler, device,
                cfg["accum_steps"], step_idx, cnn_chunk=cfg.get("cnn_chunk", 8),
            )
            ep_loss += loss
            ep_acc  += acc

            if do_update:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema:
                    ema.update(model)

            lr_dec = optimizer.param_groups[1]["lr"]
            logger.log_step(epoch + 1, step_idx + 1, loss, acc, lr_dec)

            if (step_idx + 1) % cfg["log_every"] == 0:
                lr_enc = optimizer.param_groups[0]["lr"]
                print(f"  Ep {epoch+1}/{cfg['num_epochs']} "
                      f"step {step_idx+1}/{len(train_loader)} | "
                      f"loss={loss:.4f}  acc={acc:.3f}  "
                      f"lr_dec={lr_dec:.2e}  lr_enc={lr_enc:.2e}")

        avg_loss  = ep_loss / len(train_loader)
        avg_acc   = ep_acc  / len(train_loader)
        val_model = ema.get_model() if ema else model
        val_loss, val_acc = validate(val_model, val_loader, criterion, device)
        elapsed   = time.time() - t0

        lr_dec = optimizer.param_groups[1]["lr"]
        lr_enc = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch+1, avg_loss, avg_acc, val_loss, val_acc, lr_dec, lr_enc, elapsed)
        print(f"Epoch {epoch+1:3d} | "
              f"train loss={avg_loss:.4f} acc={avg_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_m = ema.get_model() if ema else model
            torch.save(save_m.state_dict(), out_dir / "best_model.pt")
            print(f"  ✓ Лучшая модель сохранена (val_loss={val_loss:.4f})")

        if (epoch + 1) % cfg["save_every"] == 0:
            ckpt_data = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_loss, "best_val_loss": best_val_loss,
            }
            if ema:
                ckpt_data["ema"] = ema.shadow.state_dict()
            torch.save(ckpt_data, ckpt_path)

    print("Обучение завершено!")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, val in CONFIG.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() != "false", default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    train(vars(args))
