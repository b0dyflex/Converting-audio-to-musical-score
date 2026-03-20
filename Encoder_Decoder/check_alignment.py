"""
check_alignment.py — проверка соответствия спектрограммы и токенов
Для каждого сегмента MIDI-файла строит два графика:
  1. Mel-спектрограмма сегмента (то, что видит CNN-энкодер)
  2. Piano roll сегмента (ноты, которые токенизатор закодировал)

Если оба графика показывают одинаковый временной диапазон с
одинаковыми нотами — выравнивание корректно.

Запуск:
    python check_alignment.py --midi  examples/1.mid \\
                               --soundfont FluidR3_GM.sf2

    # Проверить конкретные сегменты:
    python check_alignment.py --midi  examples/1.mid \\
                               --soundfont FluidR3_GM.sf2 \\
                               --segments 0 1 2

    # Текстовый отчёт без графиков:
    python check_alignment.py --midi examples/1.mid --text_only
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "midi_to_fft"))

import numpy as np

from midi_to_fft import AudioConfig, MidiToFFTMap
from tokenizer import (
    MidiTokenizer, decode_token,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    _NOTE_ON_OFFSET, _NOTE_OFF_OFFSET,
    _TIME_SHIFT_OFFSET, _VELOCITY_OFFSET,
    TIME_STEP_MS,
)

try:
    import pretty_midi

    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

try:
    import matplotlib

    matplotlib.use("Agg")  # без GUI (сохраняем в файл)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ──────────────────────────────────────────────────────────────
#  Текстовый отчёт по сегментам
# ──────────────────────────────────────────────────────────────
def text_report(midi_path: str, config: AudioConfig) -> None:
    if not HAS_PRETTY_MIDI:
        print("[ОШИБКА] pip install pretty_midi")
        return

    pm = pretty_midi.PrettyMIDI(midi_path)
    seg = config.segment_size_sec
    dur = pm.get_end_time()
    n_seg = int(np.ceil(dur / seg))
    tok = MidiTokenizer(max_seq_len=512)

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    print(f"\n{'=' * 70}")
    print(f"Файл:     {Path(midi_path).name}")
    print(f"Длина:    {dur:.2f} сек  - {n_seg} сегментов по {seg} сек")
    print(f"{'=' * 70}")

    for i in range(n_seg):
        t_start = i * seg
        t_end = t_start + seg

        # Ноты в окне
        notes_in_window = []
        for inst in pm.instruments:
            for note in inst.notes:
                if t_start <= note.start < t_end:
                    name = note_names[note.pitch % 12] + str(note.pitch // 12 - 1)
                    notes_in_window.append((note.start - t_start, note.end - t_start,
                                            note.pitch, name, note.velocity))
        notes_in_window.sort()

        # Токены
        tokens = tok.encode_segment(midi_path, t_start, t_end)
        real = [t for t in tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]

        print(f"\nСегмент {i:2d}  [{t_start:.1f} – {t_end:.1f} сек]  "
              f"нот={len(notes_in_window):3d}  токенов={len(real):3d}")
        print(f"  {'Начало':>7}  {'Конец':>7}  {'Питч':>5}  {'Нота':>4}  {'Вел.':>4}")
        print(f"  {'-' * 40}")
        for (s, e, p, name, vel) in notes_in_window:
            print(f"  {s:7.3f}  {e:7.3f}  {p:5d}  {name:>4}  {vel:4d}")
        if not notes_in_window:
            print("  (нет нот в этом окне)")

        # Первые 10 токенов
        preview = [decode_token(t) for t in tokens[:12]]
        print(f"  Токены: {' '.join(preview)}")

    print(f"\n{'=' * 70}")


# ──────────────────────────────────────────────────────────────
#  Визуальная проверка (matplotlib)
# ──────────────────────────────────────────────────────────────
def visual_check(
        midi_path: str,
        config: AudioConfig,
        pipeline: MidiToFFTMap,
        segments_to_show: list,
        out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib не установлен, пропускаем графики")
        return
    if not HAS_PRETTY_MIDI:
        print("[WARN] pretty_midi не установлен, piano roll недоступен")
        return

    pm = pretty_midi.PrettyMIDI(midi_path)
    seg_sec = config.segment_size_sec
    tokenizer = MidiTokenizer(max_seq_len=512)

    # Спектрограмма всего трека
    try:
        spectrograms = pipeline.process(midi_path)  # (N, F, T)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось построить спектрограмму: {e}")
        return

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for seg_idx in segments_to_show:
        if seg_idx >= spectrograms.shape[0]:
            print(f"[WARN] Сегмент {seg_idx} не существует (всего {spectrograms.shape[0]})")
            continue

        t_start = seg_idx * seg_sec
        t_end = t_start + seg_sec

        # Данные
        spec = spectrograms[seg_idx]  # (F, T)
        tokens = tokenizer.encode_segment(midi_path, t_start, t_end)

        notes_in_win = []
        for inst in pm.instruments:
            for note in inst.notes:
                if t_start <= note.start < t_end:
                    notes_in_win.append((
                        note.start - t_start,
                        min(note.end, t_end) - t_start,
                        note.pitch, note.velocity,
                    ))

        # ── Рисуем ──────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(
            f"{Path(midi_path).name} | Сегмент {seg_idx} "
            f"[{t_start:.1f}–{t_end:.1f} сек]\n"
            f"Нот: {len(notes_in_win)}   Токенов (без служебных): "
            f"{sum(1 for t in tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN))}",
            fontsize=11,
        )

        # ── График 1: Спектрограмма ──────────────────────────
        ax1 = axes[0]
        T_frames = spec.shape[1]
        t_axis = np.linspace(0, seg_sec, T_frames)

        img = ax1.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[0, seg_sec, 0, spec.shape[0]],
            cmap="magma",
            vmin=0, vmax=1,
        )
        ax1.set_title("Mel-спектрограмма сегмента (вход CNN-энкодера)")
        ax1.set_xlabel("Время (сек)")
        ax1.set_ylabel("Mel-бин")
        plt.colorbar(img, ax=ax1, label="Нормализованная амплитуда [0,1]")

        # ── График 2: Piano roll ─────────────────────────────
        ax2 = axes[1]
        if notes_in_win:
            for (s, e, pitch, vel) in notes_in_win:
                alpha = 0.4 + 0.6 * (vel / 127.0)
                rect = mpatches.FancyBboxPatch(
                    (s, pitch - 0.4), e - s, 0.8,
                    boxstyle="round,pad=0.05",
                    facecolor="steelblue", edgecolor="navy",
                    alpha=alpha, linewidth=0.5,
                )
                ax2.add_patch(rect)
                name = note_names[pitch % 12] + str(pitch // 12 - 1)
                ax2.text(s + 0.02, pitch, name, fontsize=6, va="center", color="white")

            pitches = [n[2] for n in notes_in_win]
            ax2.set_xlim(0, seg_sec)
            ax2.set_ylim(min(pitches) - 2, max(pitches) + 2)
        else:
            ax2.text(seg_sec / 2, 60, "Нот нет в этом сегменте",
                     ha="center", va="center", fontsize=12, color="gray")
            ax2.set_xlim(0, seg_sec)
            ax2.set_ylim(40, 90)

        ax2.set_title("Piano roll (ноты, закодированные в токенах — правильный ответ модели)")
        ax2.set_xlabel("Время (сек)")
        ax2.set_ylabel("MIDI питч")

        # Добавляем вертикальную сетку каждые 0.5 сек
        for xt in np.arange(0, seg_sec + 0.1, 0.5):
            ax1.axvline(xt, color="white", alpha=0.15, linewidth=0.5)
            ax2.axvline(xt, color="gray", alpha=0.3, linewidth=0.5)

        plt.tight_layout()

        out_path = out_dir / f"alignment_seg{seg_idx:02d}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Сохранено: {out_path}")


# ──────────────────────────────────────────────────────────────
#  Точечная проверка: совпадает ли token-time с audio-time
# ──────────────────────────────────────────────────────────────
def verify_token_timing(midi_path: str, config: AudioConfig) -> None:
    """
    Для каждого сегмента проверяет:
    - Токены содержат NOTE_ON?
    - Временно́е смещение первого NOTE_ON совпадает с реальным временем ноты?
    """
    if not HAS_PRETTY_MIDI:
        return

    pm = pretty_midi.PrettyMIDI(midi_path)
    seg_sec = config.segment_size_sec
    dur = pm.get_end_time()
    n_seg = int(np.ceil(dur / seg_sec))
    tokenizer = MidiTokenizer(max_seq_len=512)

    print(f"\n{'─' * 60}")
    print("Проверка временно́й точности токенов:")
    print(f"{'─' * 60}")

    all_ok = True
    for i in range(min(n_seg, 20)):  # первые 20 сегментов
        t_start = i * seg_sec
        t_end = t_start + seg_sec

        # Находим первую ноту в окне (из MIDI)
        first_note_real = None
        for inst in pm.instruments:
            for note in inst.notes:
                if t_start <= note.start < t_end:
                    t_rel = note.start - t_start
                    if first_note_real is None or t_rel < first_note_real:
                        first_note_real = t_rel

        # Декодируем токены обратно
        tokens = tokenizer.encode_segment(midi_path, t_start, t_end)
        decoded_events = tokenizer.decode(tokens)
        first_note_tok = None
        for (t, etype, pitch, vel) in decoded_events:
            if etype == "NOTE_ON":
                first_note_tok = t
                break

        if first_note_real is None and first_note_tok is None:
            status = "OK (нет нот)"
        elif first_note_real is None or first_note_tok is None:
            status = "WARN (несоответствие)"
            all_ok = False
        else:
            diff = abs(first_note_real - first_note_tok)
            ok = diff < TIME_STEP_MS / 1000.0 * 2  # допуск 2 шага = 20 мс
            status = f"OK  (diff={diff * 1000:.1f} мс)" if ok else f"WARN (diff={diff * 1000:.1f} мс)"
            if not ok:
                all_ok = False

        print(f"  Сег {i:2d} [{t_start:.1f}–{t_end:.1f}] | {status}")

    print(f"{'─' * 60}")
    print(f"Итог: {'Все сегменты выровнены верно' if all_ok else 'Есть несоответствия — проверьте токенизатор'}")


# ──────────────────────────────────────────────────────────────
#  Точка входа
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Проверка соответствия спектрограммы и токенов"
    )
    parser.add_argument("--midi", required=True, help="Путь к .mid файлу")
    parser.add_argument("--soundfont", default=None, help="Путь к .sf2 (для спектрограммы)")
    parser.add_argument("--segments", type=int, nargs="+", default=None,
                        help="Индексы сегментов для визуализации (по умолчанию 0,1,2)")
    parser.add_argument("--out_dir", default="alignment_check",
                        help="Папка для PNG-файлов")
    parser.add_argument("--text_only", action="store_true",
                        help="Только текстовый отчёт, без графиков")
    args = parser.parse_args()

    config = AudioConfig()

    print(f"\nПараметры спектрограммы:")
    print(f"  Тип:        {'Mel' if config.use_mel else 'FFT'}")
    print(f"  n_fft:      {config.n_fft}")
    print(f"  hop_length: {config.hop_length}")
    print(f"  win_length: {config.win_length}")
    print(f"  n_mels:     {config.n_mels}")
    print(f"  fmax:       {config.fmax} Гц")
    print(f"  Сегмент:    {config.segment_size_sec} сек")
    print(f"  Фреймов/сегм: ~{int(config.sample_rate * config.segment_size_sec / config.hop_length)}")

    # Текстовый отчёт
    text_report(args.midi, config)

    # Проверка точности таймингов
    verify_token_timing(args.midi, config)

    if args.text_only:
        return

    # Визуализация
    if args.soundfont is None:
        print("\n[INFO] --soundfont не указан, визуализация недоступна")
        return

    if not HAS_MATPLOTLIB:
        print("\n[INFO] pip install matplotlib для визуализации")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = MidiToFFTMap(soundfont_path=args.soundfont, config=config)
    segs = args.segments if args.segments is not None else [0, 1, 2]

    print(f"\nСтрою графики для сегментов: {segs}")
    visual_check(args.midi, config, pipeline, segs, out_dir)
    print(f"\nГрафики сохранены в: {out_dir}/")


if __name__ == "__main__":
    main()
