"""
train.py
========
Цикл обучения модели ScoreGenerationModel.

Запуск (локально или в Google Colab):
    python train.py \
        --dataset_dir /path/to/dataset \
        --output_dir  /path/to/checkpoints

В Google Colab рекомендуется вынести параметры в CONFIG ниже и запускать
напрямую через `!python train.py`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import ScoreGenerationModel
from dataset import MidiSpectrogramDataset
from tokenizer import PAD_TOKEN, VOCAB_SIZE


# ══════════════════════════════════════════════════════════════
#  Конфигурация обучения (удобно менять здесь для Colab)
# ══════════════════════════════════════════════════════════════
CONFIG = dict(
    # Данные
    dataset_dir  = "dataset",
    output_dir   = "checkpoints",
    max_seq_len  = 512,

    # Архитектура
    d_model             = 128,
    nhead               = 8,
    num_decoder_layers  = 4,
    dim_feedforward     = 512,
    dropout             = 0.1,

    # Обучение
    batch_size   = 8,
    num_epochs   = 50,
    lr           = 1e-4,
    weight_decay = 1e-4,
    warmup_steps = 500,
    grad_clip    = 1.0,
    val_ratio    = 0.1,
    log_every    = 50,    # шагов
    save_every   = 5,     # эпох
)


# ══════════════════════════════════════════════════════════════
#  Планировщик: линейный warmup + cosine decay
# ══════════════════════════════════════════════════════════════
class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        import math
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════
def accuracy_no_pad(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Token-level accuracy (без PAD)."""
    preds = logits.argmax(dim=-1)
    mask = targets != PAD_TOKEN
    correct = (preds[mask] == targets[mask]).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  Один шаг обучения
# ══════════════════════════════════════════════════════════════
def train_step(
    model, batch, criterion, optimizer, scheduler, scaler, device, grad_clip
):
    specs, src, tgt, pad_mask = [t.to(device) for t in batch]

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(specs, src, tgt_key_padding_mask=pad_mask)
        # logits: (B, seq_len, vocab_size)
        # tgt:    (B, seq_len)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt.reshape(-1)
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

    acc = accuracy_no_pad(logits.detach(), tgt)
    return loss.item(), acc


# ══════════════════════════════════════════════════════════════
#  Валидация
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in loader:
        specs, src, tgt, pad_mask = [t.to(device) for t in batch]
        logits = model(specs, src, tgt_key_padding_mask=pad_mask)
        loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        acc    = accuracy_no_pad(logits, tgt)
        total_loss += loss.item()
        total_acc  += acc
        n += 1
    model.train()
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════
#  Главный цикл
# ══════════════════════════════════════════════════════════════
def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Данные ───────────────────────────────────────────────
    full_dataset = MidiSpectrogramDataset(
        dataset_root=cfg["dataset_dir"],
        max_seq_len=cfg["max_seq_len"],
    )
    n_val   = max(1, int(len(full_dataset) * cfg["val_ratio"]))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"Train: {n_train}  |  Val: {n_val}")

    # ── Модель ───────────────────────────────────────────────
    model = ScoreGenerationModel(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        vocab_size=VOCAB_SIZE,
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)

    # ── Потери и оптимизатор ─────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["num_epochs"]
    scheduler   = WarmupCosineScheduler(optimizer, cfg["warmup_steps"], total_steps)
    scaler      = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Опционально: загрузка чекпоинта ─────────────────────
    start_epoch = 0
    ckpt_path   = out_dir / "last.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Возобновление с эпохи {start_epoch}")

    # ── Цикл обучения ────────────────────────────────────────
    best_val_loss = float("inf")
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, cfg["num_epochs"]):
        epoch_loss = 0.0
        epoch_acc  = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            loss, acc = train_step(
                model, batch, criterion, optimizer,
                scheduler, scaler, device, cfg["grad_clip"]
            )
            epoch_loss += loss
            epoch_acc  += acc

            if (step + 1) % cfg["log_every"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{cfg['num_epochs']} "
                    f"step {step+1}/{len(train_loader)} | "
                    f"loss={loss:.4f}  acc={acc:.3f}  lr={lr_now:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        avg_acc  = epoch_acc  / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d} | "
            f"train loss={avg_loss:.4f} acc={avg_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  ✓ Лучшая модель сохранена (val_loss={val_loss:.4f})")

        # Периодическое сохранение
        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_loss,
            }, ckpt_path)

    print("Обучение завершено!")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, val in CONFIG.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    train(vars(args))
