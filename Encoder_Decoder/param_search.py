"""
hparam_search.py
================
Перебор гиперпараметров модели ScoreGenerationModel.
Каждый набор параметров обучается 50 эпох, результаты пишутся в CSV.

Процесс непрерывный: если скрипт прервать и запустить снова —
он пропустит уже завершённые конфигурации и продолжит с незавершённых.

Запуск:
    python hparam_search.py --dataset_dir ./dataset --output_dir ./hparam_results

    # Остановить в любой момент — при повторном запуске продолжит с того места
    python hparam_search.py --dataset_dir ./dataset --output_dir ./hparam_results

Сетка включает:
    fft_size  : [2048, 4096]
    seq_time  : [5, 10]  секунд на сегмент
    d_model   : [128, 256, 512, 768, 1024]
    lr_decoder: [1e-3, 5e-4]
    dropout   : [0.1, 0.2]

    Итого: 2 × 2 × 5 × 2 × 2 = 80 конфигураций, по 30 эпох каждая.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
import math
import traceback
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Предполагается, что скрипт лежит рядом с model.py / dataset.py / tokenizer.py
sys.path.insert(0, str(Path(__file__).parent))

from model import ScoreGenerationModel
from dataset import MidiSpectrogramDataset
from tokenizer import PAD_TOKEN, VOCAB_SIZE


# ══════════════════════════════════════════════════════════════
#  Сетка гиперпараметров
#  Добавляй / убирай значения по своему усмотрению
# ══════════════════════════════════════════════════════════════
HPARAM_GRID = {
    # Аудио / датасет  (2 × 2 = 4)
    "fft_size":   [2048, 4096],
    "seq_time":   [5, 10],

    # Архитектура  (5)
    "d_model":    [128, 256, 512, 768, 1024],

    # Оптимизатор  (5: комбинации lr_decoder × lr_encoder)
    # Оставляем только осмысленные пары: высокий/низкий
    "lr_decoder": [1e-3, 5e-4],
    "dropout":    [0.1, 0.2],
}

# ──────────────────────────────────────────────────────────────
#  Вычисляемые зависимые параметры
#  (автоматически получаются из fft_size / seq_time / d_model)
# ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050   # стандарт librosa / AudioConfig

def _derive_params(cfg: dict) -> dict:
    """
    Дополняет cfg зависимыми параметрами которые вычисляются из
    fft_size, seq_time и d_model. Вызывается перед каждым экспериментом.
    """
    fft   = cfg["fft_size"]
    stime = cfg["seq_time"]
    dm    = cfg["d_model"]

    hop_size       = fft // 4
    max_freq_bins  = min(fft // 2, 256)           # мел-бины ≤ n_fft/2, но не >256
    max_time_steps = math.ceil(SAMPLE_RATE * stime / hop_size) + 1
    max_seq_len    = stime * 50                    # ≈50 токенов/сек — с запасом

    # nhead: наибольший делитель d_model из кандидатов [16, 8, 4, 2, 1]
    nhead = 1
    for h in [16, 8, 4, 2, 1]:
        if dm % h == 0:
            nhead = h
            break

    # dim_feedforward = 4 × d_model (стандартное соотношение трансформера)
    dim_feedforward = dm * 4

    cfg = cfg.copy()
    cfg.update(dict(
        hop_size        = hop_size,
        max_freq_bins   = max_freq_bins,
        max_time_steps  = max_time_steps,
        max_seq_len     = max_seq_len,
        nhead           = nhead,
        dim_feedforward = dim_feedforward,
    ))
    return cfg


# Фиксированные параметры (не меняются между экспериментами)
FIXED = dict(
    # Данные
    imagenet_norm   = True,
    max_segments    = 64,
    val_ratio       = 0.1,

    # Архитектура (фикс)
    num_encoder_layers  = 1,
    num_decoder_layers  = 6,
    pretrained_encoder  = True,

    # Оптимизатор (фикс)
    lr_encoder      = 1e-4,
    weight_decay    = 1e-2,
    warmup_steps    = 200,

    # Обучение (фикс)
    num_epochs      = 30,
    batch_size      = 1,
    accum_steps     = 8,
    cnn_chunk       = 4,
    grad_clip       = 1.0,

    # EMA
    use_ema     = True,
    ema_decay   = 0.999,

    # Логирование
    log_every   = 50,
)


# ══════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════
class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))
        super().__init__(optimizer, lr_lambda)


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


def accuracy_no_pad(logits, targets):
    preds   = logits.argmax(dim=-1)
    mask    = targets != PAD_TOKEN
    correct = (preds[mask] == targets[mask]).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  Один прогон обучения
# ══════════════════════════════════════════════════════════════
def run_experiment(cfg: dict, run_dir: Path, device: torch.device) -> dict:
    """
    Обучает модель cfg["num_epochs"] эпох, возвращает итоговые метрики.
    Если run_dir/last.pt существует — продолжает с места остановки.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

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

    # Фиксируем разбиение для воспроизводимости
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=gen)

    _nw = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=_nw, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=_nw, pin_memory=(device.type=="cuda"))

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

    optimizer = torch.optim.AdamW(
        model.get_param_groups(
            encoder_lr = cfg["lr_encoder"],
            decoder_lr = cfg["lr_decoder"],
        ),
        weight_decay = cfg["weight_decay"],
    )

    total_steps = (len(train_loader) // cfg["accum_steps"]) * cfg["num_epochs"]
    scheduler   = WarmupCosineScheduler(optimizer, cfg["warmup_steps"], total_steps)
    scaler      = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)

    # ── Загрузка чекпоинта (resume) ──────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    ckpt_path     = run_dir / "last.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_acc  = ckpt.get("best_val_acc",  0.0)
        if ema and "ema" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema"])
        print(f"  [resume] с эпохи {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Открываем per-run CSV ─────────────────────────────────
    run_csv = run_dir / "epochs.csv"
    write_header = not run_csv.exists() or start_epoch == 0
    run_f   = open(run_csv, "a", newline="")
    run_w   = csv.writer(run_f)
    if write_header:
        run_w.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr_dec", "elapsed_s"])

    # ── Цикл обучения ─────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    total_train_time = 0.0

    for epoch in range(start_epoch, cfg["num_epochs"]):
        ep_loss, ep_acc = 0.0, 0.0
        t0 = time.time()

        for step_idx, batch in enumerate(train_loader):
            specs, src, tgt, pad_mask = [t.to(device) for t in batch]

            with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                logits = model(specs, src,
                               tgt_key_padding_mask=pad_mask,
                               cnn_chunk=cfg.get("cnn_chunk", 8))
                loss = criterion(logits.reshape(-1, VOCAB_SIZE),
                                 tgt.reshape(-1)) / cfg["accum_steps"]

            scaler.scale(loss).backward()
            ep_loss += loss.item() * cfg["accum_steps"]
            ep_acc  += accuracy_no_pad(logits.detach(), tgt)

            do_update = (step_idx + 1) % cfg["accum_steps"] == 0
            if do_update:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema:
                    ema.update(model)

        avg_loss = ep_loss / len(train_loader)
        avg_acc  = ep_acc  / len(train_loader)

        # Валидация
        val_model = ema.get_model() if ema else model
        val_model.eval()
        tot_vl, tot_va, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                specs, src, tgt, pad_mask = [t.to(device) for t in batch]
                logits = val_model(specs, src, tgt_key_padding_mask=pad_mask)
                tot_vl += criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)).item()
                tot_va += accuracy_no_pad(logits, tgt)
                nv += 1
        model.train()
        val_loss = tot_vl / nv
        val_acc  = tot_va / nv

        elapsed = time.time() - t0
        total_train_time += elapsed

        lr_dec = optimizer.param_groups[1]["lr"]

        run_w.writerow([epoch+1, f"{avg_loss:.6f}", f"{avg_acc:.6f}",
                         f"{val_loss:.6f}", f"{val_acc:.6f}",
                         f"{lr_dec:.8f}", f"{elapsed:.1f}"])
        run_f.flush()

        print(f"  Ep {epoch+1:3d}/{cfg['num_epochs']} | "
              f"train loss={avg_loss:.4f} acc={avg_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            save_m = ema.get_model() if ema else model
            torch.save(save_m.state_dict(), run_dir / "best_model.pt")

        # Чекпоинт для resume
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "best_val_acc":  best_val_acc,
        }
        if ema:
            ckpt_data["ema"] = ema.shadow.state_dict()
        torch.save(ckpt_data, ckpt_path)

    run_f.close()
    torch.cuda.empty_cache()

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc":  best_val_acc,
        "total_train_time_s": total_train_time,
    }


# ══════════════════════════════════════════════════════════════
#  Генератор конфигураций из сетки
# ══════════════════════════════════════════════════════════════
def generate_configs() -> list[dict]:
    keys   = list(HPARAM_GRID.keys())
    values = list(HPARAM_GRID.values())
    configs = []
    for combo in product(*values):
        hparams = dict(zip(keys, combo))
        cfg = _derive_params({**FIXED, **hparams})
        configs.append(cfg)
    return configs


# ══════════════════════════════════════════════════════════════
#  Главная функция поиска
# ══════════════════════════════════════════════════════════════
def hparam_search(dataset_dir: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    print(f"Dataset:    {dataset_dir}")
    print(f"Output dir: {output_dir}\n")

    base_dir   = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    results_csv = base_dir / "search_results.csv"

    # ── Загружаем уже выполненные конфигурации ────────────────
    done_ids: set[int] = set()
    if results_csv.exists():
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_ids.add(int(row["run_id"]))
        print(f"Уже выполнено: {len(done_ids)} конфигураций — пропускаем\n")

    # ── Заголовок CSV (если файл новый) ───────────────────────
    if not results_csv.exists():
        with open(results_csv, "w", newline="") as f:
            all_hkeys = list(HPARAM_GRID.keys())
            derived_keys = ["nhead", "dim_feedforward", "max_freq_bins",
                            "max_time_steps", "max_seq_len", "hop_size"]
            csv.writer(f).writerow(
                ["run_id", "status",
                 "best_val_loss", "best_val_acc", "total_train_time_s",
                 "error"]
                + all_hkeys
                + derived_keys
            )

    # ── Все конфигурации ─────────────────────────────────────
    configs = generate_configs()
    total   = len(configs)
    print(f"Всего конфигураций: {total}")
    print(f"Осталось запустить: {total - len(done_ids)}\n")
    print("=" * 65)

    for run_id, cfg in enumerate(configs):
        if run_id in done_ids:
            continue

        cfg["dataset_dir"] = dataset_dir

        # Человекочитаемое имя конфигурации
        hkeys     = list(HPARAM_GRID.keys())
        short_str = "  ".join(f"{k}={cfg[k]}" for k in hkeys)
        derived   = (f"→ nhead={cfg['nhead']}  ffn={cfg['dim_feedforward']}  "
                     f"freq_bins={cfg['max_freq_bins']}  "
                     f"time_steps={cfg['max_time_steps']}  "
                     f"seq_len={cfg['max_seq_len']}")
        run_dir   = base_dir / f"run_{run_id:04d}"

        print(f"\n{'─'*65}")
        print(f"[{run_id+1}/{total}] RUN {run_id:04d}")
        print(f"  {short_str}")
        print(f"  {derived}")
        print(f"  dir: {run_dir}")

        status = "ok"
        result = {"best_val_loss": None, "best_val_acc": None, "total_train_time_s": None}
        error_msg = ""

        try:
            result = run_experiment(cfg, run_dir, device)
            print(f"  ✓  best_val_loss={result['best_val_loss']:.4f}  "
                  f"best_val_acc={result['best_val_acc']:.3f}  "
                  f"time={result['total_train_time_s']:.0f}s")
        except KeyboardInterrupt:
            # Позволяем прервать ОДИН эксперимент и перейти к следующему
            # (двойной Ctrl+C — выход из всего поиска)
            print("\n  [!] Прервано (Ctrl+C). Переходим к следующей конфигурации...")
            status = "interrupted"
            error_msg = "KeyboardInterrupt"
        except Exception as e:
            status    = "error"
            error_msg = str(e)
            print(f"  [ERROR] {e}")
            traceback.print_exc()

        # ── Записываем строку в общий CSV ──────────────────────
        hkeys       = list(HPARAM_GRID.keys())
        derived_keys = ["nhead", "dim_feedforward", "max_freq_bins",
                        "max_time_steps", "max_seq_len", "hop_size"]
        row = [
            run_id,
            status,
            f"{result['best_val_loss']:.6f}" if result["best_val_loss"] is not None else "",
            f"{result['best_val_acc']:.6f}"  if result["best_val_acc"]  is not None else "",
            f"{result['total_train_time_s']:.0f}" if result["total_train_time_s"] is not None else "",
            error_msg,
        ] + [cfg[k] for k in hkeys] + [cfg[k] for k in derived_keys]

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Обновляем done_ids чтобы не повторять при ошибке
        done_ids.add(run_id)

        # ── Топ-3 после каждого успешного прогона ─────────────
        if status == "ok":
            _print_top(results_csv, n=3)

    print("\n" + "=" * 65)
    print("Поиск завершён!")
    print(f"Результаты: {results_csv}")
    _print_top(results_csv, n=10)


def _print_top(csv_path: Path, n: int = 5):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" and row["best_val_loss"]:
                rows.append(row)
    rows.sort(key=lambda r: float(r["best_val_loss"]))
    print(f"\n  ── Топ-{min(n, len(rows))} по best_val_loss ──")
    hkeys = list(HPARAM_GRID.keys())
    for i, r in enumerate(rows[:n]):
        params = "  ".join(f"{k}={r[k]}" for k in hkeys)
        print(f"  #{i+1}  run_{int(r['run_id']):04d}  "
              f"val_loss={float(r['best_val_loss']):.4f}  "
              f"val_acc={float(r['best_val_acc']):.3f}  |  {params}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    parser.add_argument("--dataset_dir", required=True,
                        help="Путь к датасету (папка с sample_*/)")
    parser.add_argument("--output_dir",  default="hparam_results",
                        help="Папка для хранения всех прогонов и итогового CSV")
    args = parser.parse_args()

    try:
        hparam_search(args.dataset_dir, args.output_dir)
    except KeyboardInterrupt:
        print("\n\nПоиск прерван пользователем. Возобнови командой:")
        print(f"  python hparam_search.py --dataset_dir {args.dataset_dir} --output_dir {args.output_dir}")