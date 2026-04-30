"""
pipeline.py — Полный пайплайн: аудио → MIDI → MusicXML → PDF

Два режима работы:
  1. Папка с wav+midi парами  → обучение модели, затем конвертация в PDF
  2. Один wav файл            → транскрипция моделью → MusicXML → PDF

Структура проекта:
    Converting-audio-to-musical-score/
    ├── model/
    │   ├── model.py          ← модель транскрипции аудио → MIDI
    │   └── pipeline.py       ← этот файл
    ├── midi-musicxml/
    │   ├── converter.py      ← MIDI → MusicXML
    │   └── utils.py
    └── musicxml-pdf/
        ├── lilypond.py       ← MusicXML → PDF
        └── note_analyzer.py

Формат папки с данными (--data-dir):
    my_dataset/
    ├── song1.wav   + song1.mid   ← пара (одинаковое имя)
    ├── song2.wav   + song2.mid
    └── ...
    Wav и MIDI файлы должны иметь одинаковое имя — только расширение разное.
"""

import os
import sys
import importlib.util
from pathlib import Path


# ==============================================================
#  Визуализация обучения
# ==============================================================

def plot_training(
    train_losses: list,
    val_losses: list,
    save_dir: str,
    best_epoch: int = None,
    best_loss: float = None,
    train_accs: list = None,
    val_accs: list = None,
    train_f1s: list = None,
    val_f1s: list = None,
    train_perplexities: list = None,
    val_perplexities: list = None,
    train_top5_accs: list = None,
    val_top5_accs: list = None,
):
    """
    Строит дашборд обучения и сохраняет training_plot.png в save_dir.

    Компоновка (3 строки × 3 столбца):
      [0,0] Train vs Val Loss     [0,1] Log Loss        [0,2] Δ Loss
      [1,0] Accuracy              [1,1] F1 Score        [1,2] Perplexity
      [2,0] Top-5 Accuracy        [2,1] Скользящее ср.  [2,2] Карточка лучшей эпохи
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("⚠ matplotlib не установлен. Установите: pip install matplotlib")
        return None

    if not train_losses:
        print("⚠ Нет данных для построения графика.")
        return None

    epochs     = list(range(1, len(train_losses) + 1))
    has_val    = bool(val_losses) and any(v > 0 for v in val_losses)
    val_epochs = list(range(1, len(val_losses) + 1)) if has_val else []

    # Вычисляем best_loss / best_epoch если не переданы
    if best_loss is None:
        best_loss = min(val_losses) if has_val else min(train_losses)
    if best_epoch is None:
        best_epoch = int(np.argmin(val_losses if has_val else train_losses)) + 1

    def moving_avg(data, window=3):
        if len(data) < window:
            return list(data)
        return list(np.convolve(data, np.ones(window) / window, mode="valid"))

    def _ep(data):
        """Эпохи для произвольного списка (может быть None или []). """
        return list(range(1, len(data) + 1)) if data else []

    # ── Палитра ──────────────────────────────────────────────────────
    BG_MAIN  = "#0d0d1a"
    BG_PANEL = "#13132b"
    BG_CARD  = "#1a1a3a"
    C_TRAIN  = "#4fc3f7"
    C_VAL    = "#f48fb1"
    C_MA     = "#fff176"
    C_BEST   = "#69f0ae"
    C_POS    = "#ef9a9a"
    C_NEG    = "#a5d6a7"
    C_GRID   = "#1e1e3f"
    C_TEXT   = "#e0e0e0"
    C_MUTED  = "#888899"

    # ── Фигура 3×3 ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG_MAIN)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.32,
                            left=0.06, right=0.97, top=0.92, bottom=0.06)

    def style(ax, title, xlabel="Эпоха", ylabel=""):
        ax.set_facecolor(BG_PANEL)
        ax.set_title(title, color=C_TEXT, fontsize=10, pad=6, fontweight="bold")
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=C_MUTED, fontsize=8)
        ax.tick_params(colors=C_TEXT, labelsize=7.5)
        ax.grid(color=C_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)

    def best_vline(ax):
        if best_epoch:
            ax.axvline(best_epoch, color=C_BEST, linestyle=":", linewidth=1.6,
                       alpha=0.85, label=f"Best ep.{best_epoch}")

    leg_kw = dict(fontsize=7.5, facecolor=BG_CARD, labelcolor=C_TEXT,
                  edgecolor=C_GRID, framealpha=0.9)

    # ── [0,0] Train vs Val Loss ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style(ax, "Train vs Validation Loss", ylabel="Loss")
    ax.plot(epochs, train_losses, color=C_TRAIN, linewidth=2,
            marker="o", markersize=3, label="Train")
    if has_val:
        ax.plot(val_epochs, val_losses, color=C_VAL, linewidth=2,
                marker="s", markersize=3, linestyle="--", label="Val")
    best_vline(ax)
    ax.legend(**leg_kw)

    # ── [0,1] Log-Loss ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style(ax, "Loss (лог-масштаб)", ylabel="log(Loss)")
    ax.plot(epochs, np.log(np.clip(train_losses, 1e-9, None)),
            color=C_TRAIN, linewidth=2, marker="o", markersize=3, label="log Train")
    if has_val:
        ax.plot(val_epochs, np.log(np.clip(val_losses, 1e-9, None)),
                color=C_VAL, linewidth=2, marker="s", markersize=3,
                linestyle="--", label="log Val")
    best_vline(ax)
    ax.legend(**leg_kw)

    # ── [0,2] Δ Loss ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style(ax, "Изменение Train Loss (Δ)", ylabel="ΔLoss")
    if len(train_losses) > 1:
        d      = [train_losses[i] - train_losses[i - 1] for i in range(1, len(train_losses))]
        d_ep   = epochs[1:]
        bcolors = [C_POS if v > 0 else C_NEG for v in d]
        ax.bar(d_ep, d, color=bcolors, alpha=0.85, width=0.65)
        ax.axhline(0, color=C_TEXT, linewidth=0.8)
        pos_p = mpatches.Patch(color=C_POS, label="Рост")
        neg_p = mpatches.Patch(color=C_NEG, label="Снижение")
        ax.legend(handles=[pos_p, neg_p], **leg_kw)

    # ── [1,0] Accuracy ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style(ax, "Token Accuracy", ylabel="Accuracy")
    if train_accs:
        ax.plot(_ep(train_accs), train_accs, color=C_TRAIN, linewidth=2,
                marker="o", markersize=3, label="Train")
    if val_accs and any(v > 0 for v in val_accs):
        ax.plot(_ep(val_accs), val_accs, color=C_VAL, linewidth=2,
                marker="s", markersize=3, linestyle="--", label="Val")
    best_vline(ax)
    ax.set_ylim(0, 1)
    ax.legend(**leg_kw)

    # ── [1,1] F1 Score ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style(ax, "F1 Score (macro)", ylabel="F1")
    if train_f1s:
        ax.plot(_ep(train_f1s), train_f1s, color=C_TRAIN, linewidth=2,
                marker="o", markersize=3, label="Train")
    if val_f1s and any(v > 0 for v in val_f1s):
        ax.plot(_ep(val_f1s), val_f1s, color=C_VAL, linewidth=2,
                marker="s", markersize=3, linestyle="--", label="Val")
    best_vline(ax)
    ax.set_ylim(0, 1)
    ax.legend(**leg_kw)

    # ── [1,2] Perplexity ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    style(ax, "Perplexity", ylabel="PPL")
    if train_perplexities:
        ax.plot(_ep(train_perplexities), train_perplexities, color=C_TRAIN, linewidth=2,
                marker="o", markersize=3, label="Train")
    if val_perplexities and any(v > 1 for v in val_perplexities):
        ax.plot(_ep(val_perplexities), val_perplexities, color=C_VAL, linewidth=2,
                marker="s", markersize=3, linestyle="--", label="Val")
    best_vline(ax)
    ax.legend(**leg_kw)

    # ── [2,0] Top-5 Accuracy ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    style(ax, "Top-5 Token Accuracy", ylabel="Top-5 Acc")
    if train_top5_accs:
        ax.plot(_ep(train_top5_accs), train_top5_accs, color=C_TRAIN, linewidth=2,
                marker="o", markersize=3, label="Train")
    if val_top5_accs and any(v > 0 for v in val_top5_accs):
        ax.plot(_ep(val_top5_accs), val_top5_accs, color=C_VAL, linewidth=2,
                marker="s", markersize=3, linestyle="--", label="Val")
    best_vline(ax)
    ax.set_ylim(0, 1)
    ax.legend(**leg_kw)

    # ── [2,1] Скользящее среднее loss ────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    style(ax, "Loss — скользящее среднее (окно=3)", ylabel="Loss")
    ax.plot(epochs, train_losses, color=C_TRAIN, linewidth=1, alpha=0.3, label="Train (raw)")
    ma    = moving_avg(train_losses)
    ma_ep = list(range(2, len(ma) + 2))
    ax.plot(ma_ep, ma, color=C_MA, linewidth=2.5, label="MA Train")
    if has_val and len(val_losses) >= 3:
        val_ma    = moving_avg(val_losses)
        val_ma_ep = list(range(2, len(val_ma) + 2))
        ax.plot(val_ma_ep, val_ma, color=C_VAL, linewidth=2, linestyle="--", label="MA Val")
    best_vline(ax)
    ax.legend(**leg_kw)

    # ── [2,2] Карточка «Лучшая эпоха» ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(BG_CARD)
    ax6.axis("off")
    for sp in ax6.spines.values():
        sp.set_edgecolor(C_GRID)

    ax6.text(0.5, 0.97, "Лучшая эпоха", transform=ax6.transAxes,
             ha="center", va="top", fontsize=11, fontweight="bold", color=C_BEST)
    ax6.text(0.5, 0.80, str(best_epoch), transform=ax6.transAxes,
             ha="center", va="top", fontsize=36, fontweight="bold", color=C_BEST)
    ax6.plot([0.05, 0.95], [0.66, 0.66], color=C_GRID, linewidth=1,
             transform=ax6.transAxes, clip_on=False)

    # Сводная таблица метрик
    card_rows = [("Best loss", f"{best_loss:.4f}")]
    if train_accs:
        card_rows.append(("Final train acc", f"{train_accs[-1]:.4f}"))
    if val_accs and any(v > 0 for v in val_accs):
        card_rows.append(("Final val acc",   f"{val_accs[-1]:.4f}"))
    if train_f1s:
        card_rows.append(("Final train F1",  f"{train_f1s[-1]:.4f}"))
    if val_f1s and any(v > 0 for v in val_f1s):
        card_rows.append(("Final val F1",    f"{val_f1s[-1]:.4f}"))
    if train_perplexities:
        card_rows.append(("Final PPL",       f"{train_perplexities[-1]:.2f}"))
    if len(train_losses) > 1:
        pct = (train_losses[0] - train_losses[-1]) / (abs(train_losses[0]) + 1e-9) * 100
        card_rows.append(("Loss улучшение",  f"{pct:+.1f}%"))
    card_rows.append(("Эпох обучено", str(len(train_losses))))

    y_start = 0.63
    row_h   = min(0.08, 0.60 / max(len(card_rows), 1))
    for i, (label, value) in enumerate(card_rows):
        y  = y_start - i * row_h
        bg = "#111128" if i % 2 == 0 else BG_CARD
        ax6.add_patch(plt.Rectangle((0.02, y - row_h * 0.45), 0.96, row_h,
                                    transform=ax6.transAxes,
                                    facecolor=bg, edgecolor="none", zorder=0))
        ax6.text(0.07, y, label, transform=ax6.transAxes,
                 ha="left", va="center", fontsize=8.5, color=C_MUTED)
        ax6.text(0.93, y, value, transform=ax6.transAxes,
                 ha="right", va="center", fontsize=8.5, fontweight="bold",
                 color=C_BEST if ("loss" in label.lower() or "%" in value) else C_TEXT)

    # ── Общий заголовок ───────────────────────────────────────────────
    fig.suptitle("Audio → Music  ·  Визуализация обучения",
                 color="#ffffff", fontsize=14, fontweight="bold", y=0.975)

    save_path = os.path.join(save_dir, "training_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close(fig)

    print(f"✓ График обучения сохранён: {save_path}")
    return save_path


# ==============================================================
#  НАСТРОЙКА ПУТЕЙ — измените только эти строки
# ==============================================================

# Корневая папка проекта
BASE_DIR = r"C:\Users\badim\PycharmProjects\Converting-audio-to-musical-score"

# Путь к исполняемому файлу LilyPond
LILYPOND_EXE = r"C:\Program Files\LilyPond\bin\lilypond.exe"   # ← укажите свой путь

# Путь к сохранённой / сохраняемой модели (.pth)
MODEL_PATH = r"C:\Users\badim\PycharmProjects\Converting-audio-to-musical-score\model\audio2music_best.pth"

# ==============================================================
#  Пути к папкам — формируются автоматически из BASE_DIR
# ==============================================================

DIR_MODEL         = os.path.join(BASE_DIR, "model")
DIR_MIDI_MUSICXML = os.path.join(BASE_DIR, "midi-musicxml")
DIR_MUSICXML_PDF  = os.path.join(BASE_DIR, "musicxml-pdf")

# ==============================================================


def _load_module(name: str, filepath: str):
    """Динамически загружает Python-модуль из указанного пути."""
    module_dir = os.path.dirname(filepath)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None:
        raise ImportError(f"Не удалось найти модуль по пути: {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ==============================================================
#  Сканирование папки с данными
# ==============================================================

def scan_data_dir(data_dir: str, recursive: bool = False):
    """
    Сканирует папку и возвращает:
      - wav_midi_pairs : список (wav_path, midi_path) — для обучения
      - midi_only      : список midi_path — есть MIDI, нет WAV
      - wav_only       : список wav_path  — есть WAV, нет MIDI

    Пары определяются по совпадению имени файла (без расширения).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {data_dir}")

    pattern = "**/*" if recursive else "*"

    wav_files  = {p.stem: p for p in data_dir.glob(pattern) if p.suffix.lower() in ('.wav', '.mp3', '.flac')}
    midi_files = {p.stem: p for p in data_dir.glob(pattern) if p.suffix.lower() in ('.mid', '.midi')}

    common      = set(wav_files) & set(midi_files)
    wav_midi_pairs = [(wav_files[s], midi_files[s]) for s in sorted(common)]
    midi_only      = [midi_files[s] for s in sorted(set(midi_files) - common)]
    wav_only       = [wav_files[s]  for s in sorted(set(wav_files)  - common)]

    print(f"\nСканирование папки: {data_dir}")
    print(f"  Пар wav+midi : {len(wav_midi_pairs)}")
    print(f"  Только MIDI  : {len(midi_only)}")
    print(f"  Только WAV   : {len(wav_only)}")

    return wav_midi_pairs, midi_only, wav_only


# ==============================================================
#  Обучение модели на парах wav+midi
# ==============================================================

def train_on_data_dir(
    data_dir: str,
    model_save_path: str = MODEL_PATH,
    recursive: bool = False,
    batch_size: int = 4,
    epochs: int = None,       # None = автоподбор по размеру датасета
    max_samples: int = 20000,
):
    """
    Обучает модель на папке с wav+midi парами.

    Параметры
    ----------
    data_dir         : папка с .wav и .mid файлами (одинаковые имена)
    model_save_path  : куда сохранить обученную модель (.pth)
    recursive        : искать файлы в поддиректориях
    batch_size       : размер батча
    epochs           : количество эпох (None = автоподбор)
    max_samples      : максимум пар для обучения

    Возвращает
    ----------
    Путь к лучшей сохранённой модели.
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 60)

    wav_midi_pairs, midi_only, wav_only = scan_data_dir(data_dir, recursive)

    if not wav_midi_pairs:
        raise ValueError(
            f"В папке '{data_dir}' не найдено пар wav+midi.\n"
            f"Убедитесь, что wav и midi файлы имеют одинаковые имена:\n"
            f"  song1.wav + song1.mid\n"
            f"  song2.wav + song2.mid"
        )

    if wav_only:
        print(f"\n⚠ {len(wav_only)} WAV файлов без пары MIDI — будут пропущены при обучении:")
        for p in wav_only[:5]:
            print(f"   {p.name}")
        if len(wav_only) > 5:
            print(f"   ... и ещё {len(wav_only) - 5}")

    if midi_only:
        print(f"\n⚠ {len(midi_only)} MIDI файлов без пары WAV — будут конвертированы напрямую в PDF:")
        for p in midi_only[:5]:
            print(f"   {p.name}")
        if len(midi_only) > 5:
            print(f"   ... и ещё {len(midi_only) - 5}")

    # Загружаем model.py
    model_module = _load_module("model", os.path.join(DIR_MODEL, "model.py"))

    # Строим временную папку с данными в нужном формате
    # (model.py ожидает папку, где лежат .mid и .wav с одинаковыми именами)
    save_dir = str(Path(model_save_path).parent)

    train_loader, val_loader, tokenizer = model_module.prepare_training_data_gpu(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        max_samples=max_samples,
    )

    if len(train_loader.dataset) == 0:
        raise ValueError("Датасет пуст — обучение невозможно.")

    dataset_size = len(train_loader.dataset)
    if epochs is None:
        # Автоподбор: больше данных → больше эпох нужно для сходимости
        if dataset_size < 100:
            epochs = 30
        elif dataset_size < 500:
            epochs = 50
        elif dataset_size < 2000:
            epochs = 75
        else:
            epochs = 100
    print(f"\nДатасет: {dataset_size} пар | Эпох: {epochs}")

    # Размер модели по GPU
    import torch
    device = model_module.device
    vocab_size = max(100, tokenizer.vocab_size)

    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        hidden_dim     = 512 if gpu_mem >= 8 else (384 if gpu_mem >= 4 else 256)
        enc_layers     = 4   if gpu_mem >= 8 else (3   if gpu_mem >= 4 else 2)
        dec_layers     = 6   if gpu_mem >= 8 else (4   if gpu_mem >= 4 else 3)
    else:
        hidden_dim, enc_layers, dec_layers = 256, 2, 3

    model = model_module.Audio2MusicModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
    )
    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    trainer = model_module.Audio2MusicTrainer(model, tokenizer, device)
    best_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir,
    )

    best_model_path = os.path.join(save_dir, "audio2music_best.pth")
    print(f"\n✓ Обучение завершено. Лучший loss: {best_loss:.4f}")
    print(f"✓ Модель сохранена: {best_model_path}")

    # Берём best_epoch прямо из тренера (сохраняется при каждом улучшении)
    val_losses_list   = trainer.val_losses   if trainer.val_losses   else []
    train_losses_list = trainer.train_losses if trainer.train_losses else []
    best_epoch        = trainer.best_epoch   if trainer.best_epoch   else None

    # Строим и сохраняем графики
    print("\nГенерация графиков обучения...")
    plot_training(
        train_losses=train_losses_list,
        val_losses=val_losses_list,
        save_dir=save_dir,
        best_epoch=best_epoch,
        best_loss=best_loss,
        train_accs=getattr(trainer, 'train_accs', None),
        val_accs=getattr(trainer, 'val_accs', None),
        train_f1s=getattr(trainer, 'train_f1s', None),
        val_f1s=getattr(trainer, 'val_f1s', None),
        train_perplexities=getattr(trainer, 'train_perplexities', None),
        val_perplexities=getattr(trainer, 'val_perplexities', None),
        train_top5_accs=getattr(trainer, 'train_top5_accs', None),
        val_top5_accs=getattr(trainer, 'val_top5_accs', None),
    )

    return best_model_path, midi_only


# ==============================================================
#  Транскрипция одного wav файла
# ==============================================================

def transcribe_wav(
    wav_path: str,
    output_dir: str,
    output_name: str = None,
    model_path: str = MODEL_PATH,
) -> tuple:
    """
    Транскрибирует wav → MIDI + MusicXML через обученную модель.
    Возвращает (midi_path, musicxml_path).
    """
    if output_name is None:
        output_name = Path(wav_path).stem

    midi_path     = os.path.join(output_dir, f"{output_name}.mid")
    musicxml_path = os.path.join(output_dir, f"{output_name}.musicxml")

    model_module = _load_module("model", os.path.join(DIR_MODEL, "model.py"))
    inferencer   = model_module.Audio2MusicInference(model_path=model_path)

    midi_out, musicxml_out = inferencer.transcribe_audio(
        wav_path,
        output_midi_path=midi_path,
        output_musicxml_path=musicxml_path,
    )
    return midi_out, musicxml_out


# ==============================================================
#  MIDI → MusicXML → PDF
# ==============================================================

def midi_to_pdf(
    midi_path: str,
    output_dir: str,
    output_name: str = None,
    lilypond_path: str = LILYPOND_EXE,
    include_piano_roll: bool = False,
    include_metadata: bool = True,
    scale_factor: float = 0.8,
) -> dict:
    """
    Конвертирует MIDI → улучшенный MusicXML → PDF.
    Возвращает dict с ключами 'musicxml', 'pdf'.
    """
    if output_name is None:
        output_name = Path(midi_path).stem

    musicxml_path = os.path.join(output_dir, f"{output_name}.musicxml")
    pdf_path      = os.path.join(output_dir, f"{output_name}.pdf")

    # ШАГ: MIDI → MusicXML
    converter_module = _load_module("converter", os.path.join(DIR_MIDI_MUSICXML, "converter.py"))
    midi_converter   = converter_module.MidiToMusicXMLConverter()
    musicxml_result  = midi_converter.convert_single_file(midi_path, musicxml_path)

    if not musicxml_result:
        raise RuntimeError(f"Не удалось конвертировать MIDI → MusicXML: {midi_path}")

    # ШАГ: MusicXML → PDF
    lilypond_module = _load_module("lilypond", os.path.join(DIR_MUSICXML_PDF, "lilypond.py"))
    pdf_converter   = lilypond_module.MusicXMLtoPDFConverter(lilypond_path)
    pdf_result      = pdf_converter.convert_file(
        musicxml_result,
        pdf_path,
        scale_factor=scale_factor,
        include_metadata=include_metadata,
        include_piano_roll=include_piano_roll,
    )

    return {"musicxml": musicxml_result, "pdf": pdf_result}


# ==============================================================
#  Главный пайплайн
# ==============================================================

def run_pipeline(
    audio_path: str = None,
    data_dir: str = None,
    output_dir: str = ".",
    output_name: str = None,
    lilypond_path: str = LILYPOND_EXE,
    model_path: str = MODEL_PATH,
    train: bool = False,
    recursive: bool = False,
    include_piano_roll: bool = False,
    include_metadata: bool = True,
    scale_factor: float = 0.8,
    batch_size: int = 4,
    epochs: int = None,
    max_samples: int = 20000,
) -> dict:
    """
    Универсальный пайплайн. Три сценария:

    1. data_dir + train=True
       Обучает модель на парах wav+midi из папки.
       MIDI-only файлы конвертируются напрямую в PDF.

    2. data_dir + train=False
       Использует готовую модель для транскрипции всех wav из папки → PDF.
       MIDI-only файлы конвертируются напрямую.

    3. audio_path (один wav файл)
       Транскрибирует один файл готовой моделью → PDF.
    """

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # ----------------------------------------------------------
    # Сценарий 1 и 2: папка с wav+midi
    # ----------------------------------------------------------
    if data_dir:
        wav_midi_pairs, midi_only, wav_only = scan_data_dir(data_dir, recursive)

        # Обучение если запрошено
        if train:
            print("\n" + "=" * 60)
            print("РЕЖИМ: Обучение на папке с wav+midi")
            print("=" * 60)
            model_path, _ = train_on_data_dir(
                data_dir=data_dir,
                model_save_path=model_path,
                recursive=recursive,
                batch_size=batch_size,
                epochs=epochs,
                max_samples=max_samples,
            )
        else:
            print("\n" + "=" * 60)
            print("РЕЖИМ: Транскрипция wav из папки (готовая модель)")
            print("=" * 60)

        # Транскрибируем wav → MIDI → PDF
        if wav_midi_pairs or wav_only:
            wav_files = [p[0] for p in wav_midi_pairs] + wav_only
            print(f"\nТранскрипция {len(wav_files)} WAV файлов...")
            for i, wav_path in enumerate(wav_files, 1):
                print(f"\n[{i}/{len(wav_files)}] {wav_path.name}")
                try:
                    midi_out, musicxml_out = transcribe_wav(str(wav_path), output_dir, model_path=model_path)
                    result = midi_to_pdf(
                        midi_out, output_dir,
                        lilypond_path=lilypond_path,
                        include_piano_roll=include_piano_roll,
                        include_metadata=include_metadata,
                        scale_factor=scale_factor,
                    )
                    result["midi"] = midi_out
                    result["source"] = str(wav_path)
                    all_results.append(result)
                    print(f"  ✓ PDF: {result['pdf']}")
                except Exception as e:
                    print(f"  ✗ Ошибка: {e}")
                    all_results.append({"error": str(e), "source": str(wav_path)})

        # Конвертируем MIDI-only → PDF напрямую (без модели)
        if midi_only:
            print(f"\nПрямая конвертация {len(midi_only)} MIDI файлов...")
            for i, midi_path in enumerate(midi_only, 1):
                print(f"\n[{i}/{len(midi_only)}] {midi_path.name}")
                try:
                    result = midi_to_pdf(
                        str(midi_path), output_dir,
                        lilypond_path=lilypond_path,
                        include_piano_roll=include_piano_roll,
                        include_metadata=include_metadata,
                        scale_factor=scale_factor,
                    )
                    result["source"] = str(midi_path)
                    all_results.append(result)
                    print(f"  ✓ PDF: {result['pdf']}")
                except Exception as e:
                    print(f"  ✗ Ошибка: {e}")
                    all_results.append({"error": str(e), "source": str(midi_path)})

    # ----------------------------------------------------------
    # Сценарий 3: один wav файл
    # ----------------------------------------------------------
    elif audio_path:
        print("=" * 60)
        print(f"РЕЖИМ: Транскрипция одного файла [{Path(audio_path).name}]")
        print("=" * 60)

        print("\nШАГ 1: WAV → MIDI + MusicXML (model.py)")
        midi_out, musicxml_out = transcribe_wav(audio_path, output_dir, output_name, model_path)
        print(f"  ✓ MIDI:     {midi_out}")
        print(f"  ✓ MusicXML: {musicxml_out}")

        print("\nШАГ 2: MIDI → улучшенный MusicXML + PDF")
        result = midi_to_pdf(
            midi_out, output_dir, output_name,
            lilypond_path=lilypond_path,
            include_piano_roll=include_piano_roll,
            include_metadata=include_metadata,
            scale_factor=scale_factor,
        )
        result["midi"] = midi_out
        all_results.append(result)

    else:
        raise ValueError("Укажите audio_path (один файл) или data_dir (папка с данными).")

    # ----------------------------------------------------------
    # Итог
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("ПАЙПЛАЙН ЗАВЕРШЁН")
    print("=" * 60)
    success = sum(1 for r in all_results if "error" not in r)
    failed  = len(all_results) - success
    print(f"  Успешно: {success}  |  Ошибок: {failed}  |  Всего: {len(all_results)}")

    return {"results": all_results, "success": success, "failed": failed}


# ==============================================================
#  Запуск из командной строки
# ==============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Пайплайн: аудио → MIDI → MusicXML → PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Папка с wav+midi — только конвертация (модель уже обучена)
  python pipeline.py -d "C:\\dataset\\" -o results/

  # Папка с wav+midi — сначала обучить, потом конвертировать
  python pipeline.py -d "C:\\dataset\\" -o results/ --train

  # Один wav файл
  python pipeline.py audio.wav -o results/

  # Дополнительные опции
  python pipeline.py -d "C:\\dataset\\" --piano-roll --recursive
  python pipeline.py audio.wav --lilypond "D:\\LilyPond\\bin\\lilypond.exe"
        """,
    )

    parser.add_argument("audio",            nargs="?",            help="Путь к одному WAV файлу")
    parser.add_argument("-d", "--data-dir", default=None,         help="Папка с wav+midi файлами")
    parser.add_argument("-o", "--output-dir",   default=".",      help="Выходная папка")
    parser.add_argument("-n", "--name",         default=None,     help="Имя файла (только для одиночного режима)")
    parser.add_argument("--train",          action="store_true",  help="Обучить модель перед конвертацией")
    parser.add_argument("-r", "--recursive",action="store_true",  help="Рекурсивно обходить поддиректории")
    parser.add_argument("--base-dir",       default=BASE_DIR,     help="Корневая папка проекта")
    parser.add_argument("--lilypond",       default=LILYPOND_EXE, help="Путь к lilypond.exe")
    parser.add_argument("--model",          default=MODEL_PATH,   help="Путь к .pth-файлу модели")
    parser.add_argument("--piano-roll",     action="store_true",  help="Добавить пиано-ролл в PDF")
    parser.add_argument("--no-metadata",    action="store_true",  help="Не добавлять метаданные в PDF")
    parser.add_argument("--scale",          type=float, default=0.8, help="Масштаб страницы PDF (0.5–1.5)")
    parser.add_argument("--batch-size",     type=int,   default=4,   help="Размер батча при обучении")
    parser.add_argument("--epochs",         type=int,   default=None, help="Эпох обучения (по умолчанию автоподбор)")
    parser.add_argument("--max-samples",    type=int,   default=20000, help="Максимум пар для обучения (0 = все)")

    args = parser.parse_args()

    if args.base_dir != BASE_DIR:
        DIR_MODEL         = os.path.join(args.base_dir, "model")
        DIR_MIDI_MUSICXML = os.path.join(args.base_dir, "midi-musicxml")
        DIR_MUSICXML_PDF  = os.path.join(args.base_dir, "musicxml-pdf")

    run_pipeline(
        audio_path=args.audio,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_name=args.name,
        lilypond_path=args.lilypond,
        model_path=args.model,
        train=args.train,
        recursive=args.recursive,
        include_piano_roll=args.piano_roll,
        include_metadata=not args.no_metadata,
        scale_factor=args.scale,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_samples=args.max_samples,
    )