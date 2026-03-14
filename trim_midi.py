"""
trim_midi.py
============
Подрезает MIDI-файлы до целевой длительности.

Логика:
  - Короткие (< min_duration_sec):  пропускаются или дублируются (--duplicate_short)
  - Нормальные (в пределах target ± tolerance): копируются как есть
  - Длинные (> target + tolerance): нарезаются на куски по target секунд
    Каждый кусок сохраняется как отдельный файл: song_part00.mid, song_part01.mid ...

Запуск:
    python trim_midi.py --input_dir ./midi_raw --output_dir ./midi_trimmed

    # Кастомные параметры:
    python trim_midi.py --input_dir ./midi_raw --output_dir ./midi_trimmed \\
        --target 60 --tolerance 15 --min_duration 30
"""

import argparse
import shutil
from pathlib import Path

try:
    import pretty_midi
except ImportError:
    raise SystemExit("pip install pretty_midi")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def trim_midi_to_duration(
    pm: pretty_midi.PrettyMIDI,
    start_sec: float,
    end_sec: float,
) -> pretty_midi.PrettyMIDI:
    """
    Вырезает кусок [start_sec, end_sec) из PrettyMIDI-объекта.
    Возвращает новый PrettyMIDI с нотами только из этого окна.
    Время нот смещается: новый t=0 соответствует start_sec оригинала.
    """
    out = pretty_midi.PrettyMIDI(initial_tempo=_get_tempo(pm))

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name,
        )
        for note in inst.notes:
            # Включаем ноту если она НАЧАЛАСЬ в окне
            if start_sec <= note.start < end_sec:
                new_start = note.start - start_sec
                # Конец ноты ограничиваем концом окна
                new_end   = min(note.end, end_sec) - start_sec
                if new_end > new_start:
                    new_inst.notes.append(pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=new_start,
                        end=new_end,
                    ))
        if new_inst.notes:
            out.instruments.append(new_inst)

    return out


def _get_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    tempos = pm.get_tempo_changes()
    if len(pm.get_tempo_changes()) > 0:
        _, tempo_values = pm.get_tempo_changes(), pm._tick_scales
        # Берём начальный темп
        try:
            return pm.estimate_tempo()
        except Exception:
            return 120.0
    return 120.0


def process(
    input_dir: str,
    output_dir: str,
    target_sec: float   = 60.0,
    tolerance_sec: float = 15.0,   # ±15 сек — не режем
    min_duration_sec: float = 20.0, # короче — пропускаем
    duplicate_short: bool = False,  # дублировать короткие до target
    dry_run: bool = False,
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(
        list(input_dir.glob("*.mid")) + list(input_dir.glob("*.midi"))
    )
    if not midi_files:
        print(f"MIDI не найдены в {input_dir}")
        return

    print(f"Найдено: {len(midi_files)} файлов")
    print(f"Параметры: target={target_sec}с  tolerance=±{tolerance_sec}с  "
          f"min={min_duration_sec}с  dry_run={dry_run}\n")

    stats = {"copied": 0, "trimmed": 0, "skipped": 0, "parts": 0, "errors": 0}

    for midi_path in tqdm(midi_files, desc="Обработка"):
        try:
            pm  = pretty_midi.PrettyMIDI(str(midi_path))
            dur = pm.get_end_time()

            # ── Слишком короткий → пропускаем ────────────────
            if dur < min_duration_sec:
                stats["skipped"] += 1
                if dry_run:
                    print(f"  SKIP  {midi_path.name:40s}  {dur:.1f}s  (< {min_duration_sec}s)")
                continue

            # ── Нормальная длина → копируем как есть ─────────
            if dur <= target_sec + tolerance_sec:
                stats["copied"] += 1
                if dry_run:
                    print(f"  COPY  {midi_path.name:40s}  {dur:.1f}s")
                else:
                    shutil.copy2(midi_path, output_dir / midi_path.name)
                continue

            # ── Длинный → нарезаем на куски ──────────────────
            n_parts  = int(dur // target_sec)
            leftover = dur - n_parts * target_sec

            # Если остаток >= min_duration_sec — добавляем ещё один кусок
            parts = []
            for i in range(n_parts):
                parts.append((i * target_sec, (i + 1) * target_sec))
            if leftover >= min_duration_sec:
                parts.append((n_parts * target_sec, dur))

            stem = midi_path.stem
            if dry_run:
                print(f"  CUT   {midi_path.name:40s}  {dur:.1f}s  → {len(parts)} частей")
                for i, (s, e) in enumerate(parts):
                    print(f"         part{i:02d}: [{s:.1f}–{e:.1f}]")
            else:
                for i, (t_start, t_end) in enumerate(parts):
                    chunk = trim_midi_to_duration(pm, t_start, t_end)
                    if not any(inst.notes for inst in chunk.instruments):
                        continue   # пустой кусок — пропускаем
                    out_name = output_dir / f"{stem}_part{i:02d}.mid"
                    chunk.write(str(out_name))
                    stats["parts"] += 1

            stats["trimmed"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"  [ОШИБКА] {midi_path.name}: {e}")

    # ── Итоговая статистика ───────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Итого файлов на входе:   {len(midi_files)}")
    print(f"  Скопировано (норм.):   {stats['copied']}")
    print(f"  Нарезано треков:       {stats['trimmed']}  → {stats['parts']} частей")
    print(f"  Пропущено (коротк.):   {stats['skipped']}")
    print(f"  Ошибок:                {stats['errors']}")
    if not dry_run:
        out_files = len(list(output_dir.glob("*.mid")))
        print(f"\nФайлов в {output_dir}:  {out_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",       required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--target",          type=float, default=60.0,
                        help="Целевая длительность в секундах (default: 60)")
    parser.add_argument("--tolerance",       type=float, default=15.0,
                        help="Допуск ±сек (файлы в диапазоне не режутся, default: 15)")
    parser.add_argument("--min_duration",    type=float, default=20.0,
                        help="Короче этого — пропускается (default: 20)")
    parser.add_argument("--dry_run",         action="store_true",
                        help="Только показать что будет, не записывать файлы")
    args = parser.parse_args()

    process(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        target_sec       = args.target,
        tolerance_sec    = args.tolerance,
        min_duration_sec = args.min_duration,
        dry_run          = args.dry_run,
    )