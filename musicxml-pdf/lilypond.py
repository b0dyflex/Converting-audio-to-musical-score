import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from music21 import converter, environment, metadata, stream, meter, key, tempo, chord, harmony, dynamics, instrument, \
    note, interval, pitch, scale, analysis
from music21.stream.base import Score
import argparse
import traceback
import time
import json
from datetime import datetime
import statistics
from note_analyzer import NoteAnalyzer
from decimal import Decimal
from fractions import Fraction


class MusicJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            return {
                '__type__': 'Fraction',
                'numerator': obj.numerator,
                'denominator': obj.denominator,
                'value': float(obj)
            }
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items()
                    if not k.startswith('_')}
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif hasattr(obj, '__str__'):
            return str(obj)
        return super().default(obj)


class MusicXMLtoPDFConverter:
    DEFAULT_LILYPOND_PATH = r"C:\Program Files\LilyPond\bin\lilypond.exe"

    def __init__(self, lilypond_path: Optional[str] = None) -> None:
        self.env: Optional[environment.Environment] = None
        self.lilypond_path: Optional[str] = None

        if lilypond_path is None:
            lilypond_path = self.DEFAULT_LILYPOND_PATH

        self.setup_lilypond_environment(lilypond_path)

    def setup_lilypond_environment(self, lilypond_path: str) -> None:
        print(f"Настройка LilyPond по пути: {lilypond_path}")
        if not os.path.exists(lilypond_path):
            raise FileNotFoundError(
                f"LilyPond не найден по указанному пути: {lilypond_path}\n"
                f"Убедитесь, что LilyPond установлен по адресу: {lilypond_path}\n"
                "Или укажите правильный путь при создании конвертера."
            )
        if not os.path.isfile(lilypond_path):
            raise FileNotFoundError(
                f"Указанный путь ведёт к директории, а не к файлу: {lilypond_path}"
            )
        if sys.platform == "win32" and not lilypond_path.lower().endswith('.exe'):
            print(f"Предупреждение: Путь не заканчивается на .exe: {lilypond_path}")

        self._verify_lilypond_executable(lilypond_path)
        env = environment.Environment()
        env['lilypondPath'] = lilypond_path
        self.env = env
        self.lilypond_path = lilypond_path
        print(f"✓ LilyPond успешно настроен: {lilypond_path}")

    def _verify_lilypond_executable(self, path: str) -> None:
        try:
            normalized_path = os.path.normpath(path)
            print(f"Проверка LilyPond: {normalized_path}")
            result = subprocess.run(
                [normalized_path, '--version'],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    version_line = lines[0]
                    print(f"✓ LilyPond версия: {version_line}")
            if result.stderr:
                print(f"Предупреждение от LilyPond: {result.stderr[:200]}...")
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Таймаут при проверке LilyPond. Возможно, программа зависла.\nПроверьте установку LilyPond.")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else "Неизвестная ошибка"
            raise RuntimeError(f"Ошибка при запуске LilyPond (код {e.returncode}):\n{error_msg[:500]}")
        except FileNotFoundError:
            alternative_paths = self._get_alternative_lilypond_paths()
            alternatives_text = "\n".join([f"  - {p}" for p in alternative_paths])
            raise FileNotFoundError(
                f"Файл LilyPond не найден по пути: {path}\n\n"
                f"Возможные альтернативные расположения LilyPond:\n"
                f"{alternatives_text}\n\n"
                f"Решение:\n"
                f"1. Убедитесь, что LilyPond установлен\n"
                f"2. Укажите правильный путь: MusicXMLtoPDFConverter(r'правильный\\путь\\lilypond.exe')"
            )
        except PermissionError:
            raise PermissionError(
                f"Нет прав на выполнение файла: {path}\nПопробуйте запустить программу с правами администратора.")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка при проверке LilyPond:\n{type(e).__name__}: {str(e)}")

    def _get_alternative_lilypond_paths(self) -> List[str]:
        alternative_paths = []
        if sys.platform == "win32":
            program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
            program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
            alternative_paths = [
                f"{program_files}\\LilyPond\\bin\\lilypond.exe",
                f"{program_files}\\LilyPond\\usr\\bin\\lilypond.exe",
                f"{program_files_x86}\\LilyPond\\bin\\lilypond.exe",
                f"{program_files_x86}\\LilyPond\\usr\\bin\\lilypond.exe",
                r"C:\LilyPond\bin\lilypond.exe",
                r"C:\LilyPond\usr\bin\lilypond.exe",
            ]
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for dir_path in path_dirs:
            exe_path = os.path.join(dir_path, "lilypond.exe")
            if os.path.exists(exe_path):
                alternative_paths.append(exe_path)
        unique_paths = []
        for path in alternative_paths:
            if path not in unique_paths and os.path.exists(path):
                unique_paths.append(path)
        return unique_paths

    # ========== МЕТОДЫ ДЛЯ ПИАНО-РОЛЛА ==========
    def _collect_piano_roll_data(self, score: Score) -> List[Dict[str, Any]]:
        """
        Собирает данные для построения пиано-ролла из всего произведения.
        Возвращает список нот с абсолютным offset, длительностью, MIDI-высотой.
        """
        notes_data = []
        try:
            flat = score.flatten()
            for element in flat.notes:  # включает Note и Chord
                if isinstance(element, chord.Chord):
                    for p in element.pitches:
                        notes_data.append({
                            'offset': element.offset,
                            'duration': element.duration.quarterLength,
                            'midi': p.midi,
                            'pitch_name': p.nameWithOctave,
                            'is_chord': True
                        })
                else:
                    if not hasattr(element, 'pitch'):
                        continue
                    notes_data.append({
                        'offset': element.offset,
                        'duration': element.duration.quarterLength,
                        'midi': element.pitch.midi,
                        'pitch_name': element.pitch.nameWithOctave,
                        'is_chord': False
                    })
        except Exception as e:
            print(f"   ⚠ Ошибка при сборе данных для пиано-ролла: {e}")
        return notes_data

    def generate_piano_roll_pdf(self, score: Score, output_path: str) -> None:
        """
        Создаёт PDF с пиано-роллом (интервальной спектрограммой) нот.
        """
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import mm
            import math

            notes_data = self._collect_piano_roll_data(score)
            if not notes_data:
                print("   ⚠ Нет данных для построения пиано-ролла")
                return

            # Определяем границы
            min_offset = min(n['offset'] for n in notes_data)
            max_offset = max(n['offset'] + n['duration'] for n in notes_data)
            time_range = max_offset - min_offset
            if time_range == 0:
                time_range = 1.0

            min_midi = min(n['midi'] for n in notes_data)
            max_midi = max(n['midi'] for n in notes_data)
            midi_range = max_midi - min_midi
            if midi_range == 0:
                midi_range = 1

            # Параметры страницы
            page_width, page_height = landscape(A4)
            margin = 20 * mm
            plot_width = page_width - 2 * margin
            plot_height = page_height - 2 * margin

            x_scale = plot_width / time_range
            y_scale = plot_height / midi_range

            c = canvas.Canvas(output_path, pagesize=landscape(A4))
            c.setTitle("Piano Roll / Interval Spectrogram")

            # Белый фон
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, page_width, page_height, fill=1)

            # Оси
            c.setStrokeColorRGB(0, 0, 0)
            c.setLineWidth(1)
            c.line(margin, margin, page_width - margin, margin)
            c.line(margin, margin, margin, page_height - margin)

            # Подписи осей
            c.setFont("Helvetica", 10)
            c.drawString(page_width / 2, margin - 10, "Время (доли)")
            c.saveState()
            c.rotate(90)
            c.drawString(page_height / 2, -margin + 5, "MIDI высота")
            c.restoreState()

            # Метки по оси Y (MIDI)
            step = max(1, (max_midi - min_midi) // 10)
            for midi in range(min_midi, max_midi + 1, step):
                y = margin + (midi - min_midi) * y_scale
                c.setFont("Helvetica", 8)
                c.drawString(margin - 25, y - 3, str(midi))
                c.setLineWidth(0.5)
                c.setStrokeColorRGB(0.8, 0.8, 0.8)
                c.line(margin, y, page_width - margin, y)

            # Метки по оси X (такты)
            measures = list(score.parts[0].getElementsByClass('Measure')) if score.parts else []
            if measures:
                measure_offsets = []
                current_offset = 0.0
                for m in measures:
                    measure_offsets.append(current_offset)
                    if hasattr(m, 'duration') and m.duration.quarterLength:
                        current_offset += m.duration.quarterLength
                    else:
                        notes_in_measure = list(m.notes)
                        if notes_in_measure:
                            last_note = notes_in_measure[-1]
                            current_offset = last_note.offset + last_note.duration.quarterLength
                        else:
                            current_offset += 1.0
                for i, off in enumerate(measure_offsets):
                    if off < min_offset or off > max_offset:
                        continue
                    x = margin + (off - min_offset) * x_scale
                    c.setFont("Helvetica", 8)
                    c.drawString(x - 5, margin - 15, str(i + 1))
                    c.setLineWidth(0.5)
                    c.setStrokeColorRGB(0.8, 0.8, 0.8)
                    c.line(x, margin, x, page_height - margin)

            # Рисуем ноты
            c.setStrokeColorRGB(0, 0, 0)
            c.setLineWidth(0.2)
            for note in notes_data:
                x = margin + (note['offset'] - min_offset) * x_scale
                width = note['duration'] * x_scale
                y = margin + (note['midi'] - min_midi) * y_scale
                rect_height = 5  # пунктов
                c.setFillColorRGB(0, 0, 1, 0.5)
                c.rect(x, y - rect_height / 2, width, rect_height, fill=1, stroke=0)

            # Заголовок
            c.setFont("Helvetica-Bold", 14)
            title = score.metadata.title if score.metadata and score.metadata.title else "Piano Roll"
            c.drawString(margin, page_height - margin + 10, f"Piano Roll / Interval Spectrogram: {title}")

            c.save()
            print(f"   ✓ Пиано-ролл PDF создан: {Path(output_path).name}")

        except Exception as e:
            print(f"   ⚠ Ошибка при создании пиано-ролла PDF: {e}")
            traceback.print_exc()

    # ========== МЕТОДЫ АНАЛИЗА НОТ ==========
    def get_note_range_overlap(self, score: Score, start_measure: int = 1, end_measure: int = None) -> List[
        Dict[str, Any]]:
        """
        Возвращает интервал нот в указанном диапазоне с нахлестом (overlap).
        """
        notes_info = []

        try:
            for i, part in enumerate(score.parts):
                part_name = getattr(part, 'partName', f'Part {i + 1}')
                measures = list(part.getElementsByClass('Measure'))

                if not measures:
                    continue

                if end_measure is None:
                    end_measure = len(measures)
                else:
                    end_measure = min(end_measure, len(measures))

                for measure_idx in range(max(0, start_measure - 1), end_measure):
                    measure = measures[measure_idx]
                    measure_num = measure_idx + 1
                    measure_notes = list(measure.notes)

                    for j, n in enumerate(measure_notes):
                        try:
                            note_info = {
                                'part': part_name,
                                'measure': measure_num,
                                'position_in_measure': j,
                                'note_object': n,
                                'token': self._create_note_token(n),
                                'characteristics': self._get_note_characteristics(n),
                                'overlap_info': self._get_overlap_info(n, measure_notes, j)
                            }
                            notes_info.append(note_info)
                        except Exception as e:
                            print(f"Ошибка обработки ноты: {e}")
                            continue

            self._add_note_relations(notes_info)

        except Exception as e:
            print(f"Ошибка при получении диапазона нот: {e}")

        return notes_info

    def _create_note_token(self, note_obj: note.Note) -> str:
        """Создает уникальный токен для ноты."""
        try:
            if hasattr(note_obj, 'pitch'):
                pitch_str = str(note_obj.pitch)
            else:
                pitch_str = 'unknown'

            if hasattr(note_obj, 'duration'):
                duration = note_obj.duration.quarterLength
                duration_str = f"D{duration}"
            else:
                duration_str = "D0"

            articulations = []
            if hasattr(note_obj, 'articulations'):
                for art in note_obj.articulations:
                    articulations.append(str(art))
            articulation_str = "_".join(articulations) if articulations else "no_articulation"

            dynamics_list = []
            if hasattr(note_obj, 'expressions'):
                for expr in note_obj.expressions:
                    if isinstance(expr, dynamics.Dynamic):
                        dynamics_list.append(str(expr))
            dynamic_str = "_".join(dynamics_list) if dynamics_list else "no_dynamic"

            token = f"{pitch_str}_{duration_str}_{articulation_str}_{dynamic_str}"
            import hashlib
            token_hash = hashlib.md5(token.encode()).hexdigest()[:8]
            return f"NOTE_{token_hash}"

        except Exception as e:
            return f"ERROR_{hash(str(e)) % 10000}"

    def _get_note_characteristics(self, note_obj: note.Note) -> Dict[str, Any]:
        """Возвращает характеристики ноты."""
        characteristics = {
            'basic': {},
            'pitch_info': {},
            'rhythmic_info': {},
            'performance_info': {},
            'contextual_info': {}
        }

        try:
            characteristics['basic'] = {
                'type': note_obj.className,
                'is_rest': isinstance(note_obj, note.Rest),
                'is_chord': isinstance(note_obj, chord.Chord)
            }

            if hasattr(note_obj, 'pitch'):
                p = note_obj.pitch
                characteristics['pitch_info'] = {
                    'name': p.name,
                    'name_with_octave': p.nameWithOctave,
                    'midi_number': p.midi,
                    'frequency': round(p.frequency, 2) if hasattr(p, 'frequency') else None,
                    'octave': p.octave,
                    'step': p.step,
                    'accidental': str(p.accidental) if p.accidental else None,
                    'is_microtone': p.isMicrotonal(),
                    'spelling': self._get_pitch_spelling(p)
                }

            if hasattr(note_obj, 'duration'):
                d = note_obj.duration
                characteristics['rhythmic_info'] = {
                    'quarter_length': d.quarterLength,
                    'type': d.type,
                    'dots': d.dots,
                    'tuplets': [str(t) for t in d.tuplets] if hasattr(d, 'tuplets') else [],
                    'is_grace': d.isGrace,
                    'is_incomplete': d.incompleteFill,
                    'duration_components': self._get_duration_components(d)
                }

            characteristics['performance_info'] = {
                'articulations': [str(a) for a in note_obj.articulations] if hasattr(note_obj, 'articulations') else [],
                'expressions': [str(e) for e in note_obj.expressions] if hasattr(note_obj, 'expressions') else [],
                'lyrics': [str(l.text) for l in note_obj.lyrics] if hasattr(note_obj, 'lyrics') else [],
                'tie': str(note_obj.tie) if hasattr(note_obj, 'tie') else None,
                'stem_direction': note_obj.stemDirection if hasattr(note_obj, 'stemDirection') else None
            }

            if hasattr(note_obj, 'volume'):
                characteristics['performance_info']['volume'] = {
                    'velocity': note_obj.volume.velocity,
                    'velocity_is_relative': note_obj.volume.velocityIsRelative
                }

            characteristics['theoretical_info'] = {
                'scale_degree': self._get_scale_degree(note_obj),
                'harmonic_function': self._estimate_harmonic_function(note_obj),
                'melodic_contour': self._get_melodic_contour(note_obj)
            }

        except Exception as e:
            characteristics['error'] = str(e)

        return characteristics

    def _get_overlap_info(self, current_note: note.Note, measure_notes: List, position: int) -> Dict[str, Any]:
        """Анализирует нахлест ноты с соседними нотами."""
        overlap_info = {
            'previous_overlap': None,
            'next_overlap': None,
            'simultaneous_notes': []
        }

        try:
            if position > 0:
                prev_note = measure_notes[position - 1]
                if self._notes_overlap(prev_note, current_note):
                    overlap_info['previous_overlap'] = {
                        'note': str(prev_note),
                        'overlap_type': self._determine_overlap_type(prev_note, current_note),
                        'overlap_ratio': self._calculate_overlap_ratio(prev_note, current_note)
                    }

            if position < len(measure_notes) - 1:
                next_note = measure_notes[position + 1]
                if self._notes_overlap(current_note, next_note):
                    overlap_info['next_overlap'] = {
                        'note': str(next_note),
                        'overlap_type': self._determine_overlap_type(current_note, next_note),
                        'overlap_ratio': self._calculate_overlap_ratio(current_note, next_note)
                    }

            if position > 0:
                for i in range(position):
                    other_note = measure_notes[i]
                    if self._notes_simultaneous(other_note, current_note):
                        overlap_info['simultaneous_notes'].append({
                            'note': str(other_note),
                            'position': i,
                            'interval': self._get_interval_between_notes(other_note, current_note)
                        })

        except Exception as e:
            overlap_info['error'] = str(e)

        return overlap_info

    def _notes_overlap(self, note1: note.Note, note2: note.Note) -> bool:
        """Проверяет, перекрываются ли ноты во времени."""
        try:
            if not hasattr(note1, 'offset') or not hasattr(note2, 'offset'):
                return False
            if not hasattr(note1, 'duration') or not hasattr(note2, 'duration'):
                return False
            end1 = note1.offset + note1.duration.quarterLength
            start2 = note2.offset
            return end1 > start2
        except:
            return False

    def _notes_simultaneous(self, note1: note.Note, note2: note.Note) -> bool:
        """Проверяет, звучат ли ноты одновременно."""
        try:
            if not hasattr(note1, 'offset') or not hasattr(note2, 'offset'):
                return False
            start1 = note1.offset
            start2 = note2.offset
            return abs(start1 - start2) < 0.01
        except:
            return False

    def _determine_overlap_type(self, note1: note.Note, note2: note.Note) -> str:
        """Определяет тип перекрытия нот."""
        try:
            if not self._notes_overlap(note1, note2):
                return "no_overlap"
            end1 = note1.offset + note1.duration.quarterLength
            start2 = note2.offset
            overlap_amount = end1 - start2

            if hasattr(note1, 'duration'):
                overlap_percentage = overlap_amount / note1.duration.quarterLength
                if overlap_percentage < 0.1:
                    return "slight_overlap"
                elif overlap_percentage < 0.3:
                    return "moderate_overlap"
                elif overlap_percentage < 0.7:
                    return "significant_overlap"
                else:
                    return "heavy_overlap"
            return "overlap"
        except:
            return "unknown"

    def _calculate_overlap_ratio(self, note1: note.Note, note2: note.Note) -> float:
        """Вычисляет коэффициент перекрытия."""
        try:
            if not self._notes_overlap(note1, note2):
                return 0.0
            end1 = note1.offset + note1.duration.quarterLength
            start2 = note2.offset
            overlap_amount = end1 - start2
            duration1 = note1.duration.quarterLength
            duration2 = note2.duration.quarterLength
            return overlap_amount / min(duration1, duration2)
        except:
            return 0.0

    def _get_interval_between_notes(self, note1: note.Note, note2: note.Note) -> Dict[str, Any]:
        """Возвращает интервал между двумя нотами."""
        interval_info = {
            'simple_name': None,
            'compound_name': None,
            'semitones': None,
            'direction': None,
            'diatonic': None,
            'chromatic': None
        }

        try:
            if hasattr(note1, 'pitch') and hasattr(note2, 'pitch'):
                intv = interval.Interval(noteStart=note1.pitch, noteEnd=note2.pitch)
                interval_info['simple_name'] = intv.simpleName
                interval_info['compound_name'] = intv.name
                interval_info['semitones'] = intv.semitones
                interval_info['direction'] = intv.direction.name if hasattr(intv.direction, 'name') else str(
                    intv.direction)
                interval_info['diatonic'] = intv.diatonic.name if hasattr(intv.diatonic, 'name') else str(intv.diatonic)
                interval_info['chromatic'] = intv.chromatic.name if hasattr(intv.chromatic, 'name') else str(
                    intv.chromatic)
                interval_info['is_consonant'] = intv.isConsonant()
                interval_info['is_dissonant'] = intv.isDissonant()
                interval_info['is_perfect'] = intv.isPerfectType
                interval_info['generic'] = intv.generic.simpleUndirected
        except Exception as e:
            interval_info['error'] = str(e)

        return interval_info

    def _add_note_relations(self, notes_info: List[Dict[str, Any]]) -> None:
        """Добавляет информацию о связях между нотами."""
        for i in range(len(notes_info)):
            current_note = notes_info[i]

            if i > 0:
                prev_note = notes_info[i - 1]
                current_note['relations'] = current_note.get('relations', {})
                current_note['relations']['previous'] = {
                    'token': prev_note['token'],
                    'interval': self._get_interval_between_notes(
                        prev_note['note_object'],
                        current_note['note_object']
                    ) if hasattr(prev_note['note_object'], 'pitch') else None,
                    'time_gap': self._calculate_time_gap(
                        prev_note['note_object'],
                        current_note['note_object']
                    )
                }

            if i < len(notes_info) - 1:
                next_note = notes_info[i + 1]
                current_note['relations'] = current_note.get('relations', {})
                current_note['relations']['next'] = {
                    'token': next_note['token'],
                    'interval': self._get_interval_between_notes(
                        current_note['note_object'],
                        next_note['note_object']
                    ) if hasattr(current_note['note_object'], 'pitch') else None
                }

    def _calculate_time_gap(self, note1: note.Note, note2: note.Note) -> float:
        """Вычисляет временной промежуток между нотами."""
        try:
            if hasattr(note1, 'offset') and hasattr(note2, 'offset'):
                end1 = note1.offset + note1.duration.quarterLength
                start2 = note2.offset
                return max(0, start2 - end1)
        except:
            pass
        return 0.0

    def _get_pitch_spelling(self, pitch_obj: pitch.Pitch) -> Dict[str, Any]:
        """Возвращает информацию о написании высоты звука."""
        spelling = {
            'enharmonic_spellings': [],
            'preferred_spelling': None,
            'spelling_ambiguity': False
        }

        try:
            enharmonics = pitch_obj.getAllCommonEnharmonics()
            spelling['enharmonic_spellings'] = [str(p) for p in enharmonics]
            spelling['preferred_spelling'] = pitch_obj.nameWithOctave
            spelling['spelling_ambiguity'] = len(enharmonics) > 1
        except:
            pass

        return spelling

    def _get_duration_components(self, duration_obj) -> List[Dict[str, Any]]:
        """Разбивает длительность на компоненты."""
        components = []

        try:
            if hasattr(duration_obj, 'tuplets') and duration_obj.tuplets:
                for tuplet in duration_obj.tuplets:
                    components.append({
                        'type': 'tuplet',
                        'ratio': str(tuplet.tupletRatio()),
                        'number_notes': tuplet.numberNotesActual,
                        'duration_notes': tuplet.numberNotesNormal
                    })

            components.append({
                'type': 'base',
                'duration': duration_obj.quarterLength,
                'type_name': duration_obj.type,
                'dots': duration_obj.dots,
                'is_grace': duration_obj.isGrace
            })

        except:
            pass

        return components

    def _get_scale_degree(self, note_obj: note.Note) -> Optional[int]:
        """Определяет ступень лада для ноты."""
        try:
            if hasattr(note_obj, 'pitch'):
                p = note_obj.pitch
                midi_num = p.midi
                scale_notes = [0, 2, 4, 5, 7, 9, 11]
                degree = (midi_num % 12)
                if degree in scale_notes:
                    return scale_notes.index(degree) + 1
        except:
            pass
        return None

    def _estimate_harmonic_function(self, note_obj: note.Note) -> str:
        """Оценивает гармоническую функцию ноты."""
        return "undefined"

    def _get_melodic_contour(self, note_obj: note.Note) -> str:
        """Определяет мелодический контур."""
        return "stable"

    def get_note_subset_by_criteria(self, notes_info: List[Dict[str, Any]],
                                    criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Возвращает подмножество нот по заданным критериям.
        """
        filtered_notes = []
        for note_info in notes_info:
            if self._meets_criteria(note_info, criteria):
                filtered_notes.append(note_info)
        return filtered_notes

    def _meets_criteria(self, note_info: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Проверяет, удовлетворяет ли нота критериям."""
        try:
            if 'pitch_range' in criteria:
                pitch_info = note_info['characteristics'].get('pitch_info', {})
                midi_num = pitch_info.get('midi_number')
                if midi_num is None:
                    return False
                pitch_range = criteria['pitch_range']
                if not (pitch_range.get('min', 0) <= midi_num <= pitch_range.get('max', 127)):
                    return False

            if 'duration_range' in criteria:
                rhythmic_info = note_info['characteristics'].get('rhythmic_info', {})
                duration = rhythmic_info.get('quarter_length')
                if duration is None:
                    return False
                dur_range = criteria['duration_range']
                if not (dur_range.get('min', 0) <= duration <= dur_range.get('max', float('inf'))):
                    return False

            if 'articulation_types' in criteria:
                perf_info = note_info['characteristics'].get('performance_info', {})
                articulations = perf_info.get('articulations', [])
                required_articulations = criteria['articulation_types']
                if required_articulations:
                    has_required = any(art.lower() in [a.lower() for a in articulations]
                                       for art in required_articulations)
                    if not has_required:
                        return False

            if 'dynamic_levels' in criteria:
                perf_info = note_info['characteristics'].get('performance_info', {})
                expressions = perf_info.get('expressions', [])
                required_dynamics = criteria['dynamic_levels']
                if required_dynamics:
                    has_required = any(dyn.lower() in [e.lower() for e in expressions]
                                       for dyn in required_dynamics)
                    if not has_required:
                        return False

            if 'overlap_type' in criteria:
                overlap_type = note_info.get('overlap_info', {}).get('previous_overlap', {}).get('overlap_type')
                if overlap_type not in criteria['overlap_type']:
                    return False

            if 'scale_degree' in criteria:
                theoretical_info = note_info['characteristics'].get('theoretical_info', {})
                degree = theoretical_info.get('scale_degree')
                if degree not in criteria['scale_degree']:
                    return False

            if 'measure_range' in criteria:
                measure = note_info.get('measure')
                measure_range = criteria['measure_range']
                if not (measure_range.get('start', 1) <= measure <= measure_range.get('end', float('inf'))):
                    return False

            return True

        except Exception as e:
            print(f"Ошибка при проверке критериев: {e}")
            return False

    def analyze_note_patterns(self, notes_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует паттерны в последовательности нот.
        """
        patterns = {
            'repetitions': [],
            'sequences': [],
            'intervallic_patterns': [],
            'rhythmic_patterns': [],
            'contour_patterns': []
        }

        try:
            patterns['repetitions'] = self._find_repetitions(notes_info)
            patterns['sequences'] = self._find_sequences(notes_info)
            patterns['intervallic_patterns'] = self._analyze_intervallic_patterns(notes_info)
            patterns['rhythmic_patterns'] = self._analyze_rhythmic_patterns(notes_info)
            patterns['contour_patterns'] = self._analyze_contour_patterns(notes_info)
        except Exception as e:
            patterns['error'] = str(e)

        return patterns

    def _find_repetitions(self, notes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Находит повторяющиеся последовательности нот."""
        repetitions = []
        try:
            tokens = [note['token'] for note in notes_info]
            for length in range(2, min(7, len(tokens) // 2 + 1)):
                for i in range(len(tokens) - length):
                    sequence = tokens[i:i + length]
                    for j in range(i + length, len(tokens) - length + 1):
                        if tokens[j:j + length] == sequence:
                            repetition = {
                                'sequence': sequence,
                                'positions': [i, j],
                                'length': length,
                                'notes': [
                                    {
                                        'token': notes_info[k]['token'],
                                        'characteristics': notes_info[k]['characteristics']
                                    }
                                    for k in range(i, i + length)
                                ]
                            }
                            repetitions.append(repetition)
        except Exception as e:
            print(f"Ошибка при поиске повторений: {e}")
        return repetitions

    def _find_sequences(self, notes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Находит секвенции (повторяющиеся паттерны с транспозицией)."""
        sequences = []
        # Упрощенная реализация
        return sequences

    def _analyze_intervallic_patterns(self, notes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Анализирует интервальные паттерны."""
        patterns = []
        # Упрощенная реализация
        return patterns

    def _analyze_rhythmic_patterns(self, notes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Анализирует ритмические паттерны."""
        patterns = []
        # Упрощенная реализация
        return patterns

    def _analyze_contour_patterns(self, notes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Анализирует паттерны мелодического контура."""
        patterns = []
        # Упрощенная реализация
        return patterns

    def export_note_analysis(self, notes_info: List[Dict[str, Any]],
                             output_path: str) -> None:
        """
        Экспортирует анализ нот в JSON файл.
        """
        try:
            export_data = {
                'metadata': {
                    'total_notes': len(notes_info),
                    'export_date': datetime.now().isoformat(),
                    'analysis_version': '1.0'
                },
                'notes': [],
                'summary': self._create_note_summary(notes_info),
                'patterns': self.analyze_note_patterns(notes_info)
            }

            for note_info in notes_info:
                simplified_note = {
                    'token': note_info['token'],
                    'measure': note_info['measure'],
                    'position_in_measure': note_info['position_in_measure'],
                    'part': note_info['part'],
                    'pitch': note_info['characteristics'].get('pitch_info', {}).get('name_with_octave'),
                    'midi_number': note_info['characteristics'].get('pitch_info', {}).get('midi_number'),
                    'duration': note_info['characteristics'].get('rhythmic_info', {}).get('quarter_length'),
                    'articulations': note_info['characteristics'].get('performance_info', {}).get('articulations', []),
                    'overlap_type': note_info.get('overlap_info', {}).get('previous_overlap', {}).get('overlap_type'),
                    'relations': note_info.get('relations', {})
                }
                export_data['notes'].append(simplified_note)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Анализ нот экспортирован: {output_path}")

        except Exception as e:
            print(f"Ошибка при экспорте анализа нот: {e}")

    def _create_note_summary(self, notes_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Создает сводку по анализу нот."""
        summary = {
            'pitch_statistics': {
                'range': {'min': 127, 'max': 0},
                'average': 0,
                'common_pitches': []
            },
            'duration_statistics': {
                'range': {'min': float('inf'), 'max': 0},
                'average': 0,
                'duration_distribution': {}
            },
            'articulation_statistics': {},
            'overlap_statistics': {}
        }

        try:
            midi_numbers = []
            pitch_counts = {}
            durations = []
            duration_counts = {}
            articulation_counts = {}
            overlap_counts = {}

            for note_info in notes_info:
                pitch_info = note_info['characteristics'].get('pitch_info', {})
                midi_num = pitch_info.get('midi_number')
                if midi_num is not None:
                    midi_numbers.append(midi_num)
                    pitch_name = pitch_info.get('name')
                    if pitch_name:
                        pitch_counts[pitch_name] = pitch_counts.get(pitch_name, 0) + 1

                rhythmic_info = note_info['characteristics'].get('rhythmic_info', {})
                duration = rhythmic_info.get('quarter_length')
                if duration is not None:
                    durations.append(duration)
                    dur_type = rhythmic_info.get('type', 'unknown')
                    duration_counts[dur_type] = duration_counts.get(dur_type, 0) + 1

                perf_info = note_info['characteristics'].get('performance_info', {})
                for articulation in perf_info.get('articulations', []):
                    articulation_counts[articulation] = articulation_counts.get(articulation, 0) + 1

                overlap_type = note_info.get('overlap_info', {}).get('previous_overlap', {}).get('overlap_type')
                if overlap_type:
                    overlap_counts[overlap_type] = overlap_counts.get(overlap_type, 0) + 1

            if midi_numbers:
                summary['pitch_statistics']['range']['min'] = min(midi_numbers)
                summary['pitch_statistics']['range']['max'] = max(midi_numbers)
                summary['pitch_statistics']['average'] = sum(midi_numbers) / len(midi_numbers)
                sorted_pitches = sorted(pitch_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                summary['pitch_statistics']['common_pitches'] = sorted_pitches

            if durations:
                summary['duration_statistics']['range']['min'] = min(durations)
                summary['duration_statistics']['range']['max'] = max(durations)
                summary['duration_statistics']['average'] = sum(durations) / len(durations)
                summary['duration_statistics']['duration_distribution'] = duration_counts

            summary['articulation_statistics'] = articulation_counts
            summary['overlap_statistics'] = overlap_counts

        except Exception as e:
            summary['error'] = str(e)

        return summary

    def demonstrate_note_analysis(self, score: Score):
        """Демонстрация функционала анализа нот."""
        notes_in_range = self.get_note_range_overlap(score, start_measure=1, end_measure=10)
        print(f"Найдено нот в диапазоне: {len(notes_in_range)}")

        criteria = {
            'pitch_range': {'min': 60, 'max': 72},
            'duration_range': {'min': 0.5, 'max': 2.0},
            'articulation_types': ['staccato', 'accent']
        }

        filtered_notes = self.get_note_subset_by_criteria(notes_in_range, criteria)
        print(f"Нот по критериям: {len(filtered_notes)}")

        patterns = self.analyze_note_patterns(notes_in_range)
        print(f"Найдено паттернов: {len(patterns.get('repetitions', []))} повторений")

        self.export_note_analysis(notes_in_range, "note_analysis.json")

        if notes_in_range:
            sample_note = notes_in_range[0]
            print(f"\nПример ноты:")
            print(f"  Токен: {sample_note['token']}")
            print(f"  Такт: {sample_note['measure']}")
            print(f"  Высота: {sample_note['characteristics']['pitch_info']['name_with_octave']}")
            print(f"  Длительность: {sample_note['characteristics']['rhythmic_info']['quarter_length']}")

            if sample_note['overlap_info']['previous_overlap']:
                print(f"  Перекрытие: {sample_note['overlap_info']['previous_overlap']['overlap_type']}")

        return {
            'total_notes': len(notes_in_range),
            'filtered_notes': len(filtered_notes),
            'patterns_found': sum(len(v) for v in patterns.values() if isinstance(v, list))
        }

    # ========== МЕТОДЫ АНАЛИЗА МЕТАДАННЫХ ==========
    def extract_structural_metadata(self, score: Score) -> Dict[str, Any]:
        print("   Извлечение структурных метаданных...")

        metadata = {
            "basic_info": {},
            "instrumentation_info": {},
            "harmonic_info": {},
            "structural_info": {},
            "statistical_info": {},
            "dynamic_info": {},
            "analysis_info": {},
            "form_analysis": {}
        }

        # 1. Основная информация
        if score.metadata:
            metadata["basic_info"] = {
                "title": score.metadata.title or "Без названия",
                "composer": score.metadata.composer or "Неизвестный композитор",
                "arranger": getattr(score.metadata, 'arranger', None),
                "opus_number": getattr(score.metadata, 'opusNumber', None),
                "movement_name": getattr(score.metadata, 'movementName', None),
                "movement_number": getattr(score.metadata, 'movementNumber', None),
                "date_created": str(score.metadata.dateCreated) if hasattr(score.metadata,
                                                                           'dateCreated') and score.metadata.dateCreated else None,
                "copyright": getattr(score.metadata, 'copyright', None)
            }

        # 2. Анализ инструментовки
        metadata["instrumentation_info"] = self._analyze_instrumentation(score)

        # 3. Гармонический анализ
        metadata["harmonic_info"] = self._analyze_harmony(score)

        # 4. Структурная информация
        parts_info = []
        total_measures = 0
        time_signatures = set()
        key_signatures = set()
        tempos = []

        for i, part in enumerate(score.parts):
            part_info = {
                "part_id": i + 1,
                "part_name": getattr(part, 'partName', f"Партия {i + 1}"),
                "instrument": metadata["instrumentation_info"]["detected_instruments"][i] if i < len(
                    metadata["instrumentation_info"]["detected_instruments"]) else "Неизвестный инструмент",
                "measures_count": len(part.getElementsByClass('Measure')) if hasattr(part, 'getElementsByClass') else 0,
                "tessitura": self._analyze_tessitura(part)
            }
            parts_info.append(part_info)
            total_measures = max(total_measures, part_info["measures_count"])

            try:
                flat_part = part.flatten()
                for ts in flat_part.getElementsByClass(meter.TimeSignature):
                    if hasattr(ts, 'ratioString'):
                        time_signatures.add(str(ts.ratioString))
            except:
                pass

            try:
                flat_part = part.flatten()
                for ks in flat_part.getElementsByClass(key.KeySignature):
                    key_str = str(ks)
                    if key_str:
                        key_signatures.add(key_str)
            except:
                pass

            try:
                flat_part = part.flatten()
                for t in flat_part.getElementsByClass(tempo.TempoIndication):
                    tempo_info = {"text": str(t)}
                    if hasattr(t, 'number'):
                        tempo_info["bpm"] = t.number
                    tempos.append(tempo_info)
            except:
                pass

        metadata["structural_info"] = {
            "parts_count": len(score.parts),
            "parts": parts_info,
            "total_measures": total_measures,
            "time_signatures": list(time_signatures),
            "key_signatures": list(key_signatures),
            "tempos": list({json.dumps(t, sort_keys=True): t for t in tempos}.values())
        }

        # 5. Статистическая информация
        metadata["statistical_info"] = self._analyze_statistics(score)

        # 6. Динамический анализ
        metadata["dynamic_info"] = self._analyze_dynamics(score)

        # 7. Анализ формы
        metadata["form_analysis"] = self._analyze_form_structure(score)

        # 8. Аналитическая информация
        metadata["analysis_info"] = {
            "estimated_difficulty": self._estimate_difficulty(score),
            "style_indicators": self._analyze_style(score),
            "ensemble_type": self._determine_ensemble_type(metadata["instrumentation_info"]),
            "genre_prediction": self._predict_genre(score, metadata)
        }

        # 9. Информация о конвертации
        metadata["conversion_info"] = {
            "conversion_date": datetime.now().isoformat(),
            "converter_version": "2.0",
            "lilypond_version": self.get_lilypond_info().get("version", "unknown"),
            "input_file": str(Path(score.metadata.title or "unknown").with_suffix(''))
        }

        return metadata

    def _analyze_instrumentation(self, score: Score) -> Dict[str, Any]:
        """Детальный анализ инструментовки."""
        instruments = []
        instrument_families = set()
        part_instruments = []

        for i, part in enumerate(score.parts):
            detected_instrument = self._detect_instrument(part)
            instruments.append(detected_instrument)
            part_instruments.append(detected_instrument)
            family = self._get_instrument_family(detected_instrument)
            instrument_families.add(family)

        unique_instruments = {}
        for instr in instruments:
            if instr in unique_instruments:
                unique_instruments[instr] += 1
            else:
                unique_instruments[instr] = 1

        return {
            "detected_instruments": part_instruments,
            "instrument_families": list(instrument_families),
            "instrument_distribution": unique_instruments,
            "total_instrument_types": len(unique_instruments),
            "ensemble_category": self._categorize_ensemble(instrument_families)
        }

    def _detect_instrument(self, part) -> str:
        """Определяет инструмент по различным признакам."""
        if hasattr(part, 'instrumentName') and part.instrumentName:
            return str(part.instrumentName)

        try:
            for elem in part.recurse().getElementsByClass('Instrument'):
                if hasattr(elem, 'instrumentName') and elem.instrumentName:
                    return str(elem.instrumentName)
        except:
            pass

        try:
            midi_program = None
            for elem in part.recurse():
                if hasattr(elem, 'midiProgram'):
                    midi_program = elem.midiProgram
                    break

            if midi_program is not None:
                instrument_names = {
                    0: "Acoustic Grand Piano", 24: "Acoustic Guitar (nylon)",
                    25: "Acoustic Guitar (steel)", 40: "Violin",
                    41: "Viola", 42: "Cello", 43: "Contrabass",
                    56: "Trumpet", 57: "Trombone", 58: "Tuba",
                    60: "French Horn", 61: "Brass Section",
                    73: "Flute", 74: "Piccolo", 75: "Recorder",
                    72: "Clarinet", 68: "Oboe", 69: "English Horn",
                    70: "Bassoon", 33: "Electric Bass (finger)",
                    34: "Electric Bass (pick)", 48: "String Ensemble",
                    49: "Synth Strings", 54: "Voice Oohs",
                }
                return instrument_names.get(midi_program, f"MIDI Program {midi_program}")
        except:
            pass

        try:
            notes = list(part.flatten().notes)
            if notes:
                pitches = []
                for n in notes:
                    if hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
                        pitches.append(n.pitch.midi)
                if pitches:
                    min_pitch = min(pitches)
                    max_pitch = max(pitches)

                    if max_pitch < 40:
                        return "Контрабас или Туба"
                    elif max_pitch < 60:
                        return "Виолончель или Тромбон"
                    elif max_pitch < 80:
                        return "Альт или Валторна"
                    elif max_pitch < 100:
                        return "Скрипка или Флейта"
                    else:
                        return "Пикколо или Высокие духовые"
        except:
            pass

        return "Неизвестный инструмент"

    def _get_instrument_family(self, instrument_name: str) -> str:
        """Определяет семейство инструмента по названию."""
        instrument_name_lower = instrument_name.lower()

        families = {
            "струнные": ["скрипка", "виолончель", "альт", "контрабас", "гитара", "арфа", "violin", "cello", "viola",
                         "bass", "guitar"],
            "деревянные духовые": ["флейта", "кларнет", "гобой", "фагот", "саксофон", "recorder", "flute", "clarinet",
                                   "oboe", "bassoon"],
            "медные духовые": ["труба", "тромбон", "туба", "валторна", "trumpet", "trombone", "tuba", "horn"],
            "ударные": ["барабан", "литавры", "тарелки", "треугольник", "drum", "timpani", "cymbals"],
            "клавишные": ["фортепиано", "орган", "клавесин", "синтезатор", "piano", "organ", "harpsichord"],
            "вокальные": ["голос", "сопрано", "альт", "тенор", "бас", "хор", "voice", "soprano", "alto", "tenor",
                          "bass", "choir"]
        }

        for family, keywords in families.items():
            for keyword in keywords:
                if keyword in instrument_name_lower:
                    return family

        return "другие"

    def _categorize_ensemble(self, instrument_families) -> str:
        """Категоризирует тип ансамбля."""
        families = list(instrument_families)

        if "вокальные" in families:
            if len(families) == 1:
                return "Хоровое произведение"
            else:
                return "Вокально-инструментальный ансамбль"
        elif "струнные" in families and len(families) <= 3:
            return "Камерный ансамбль"
        elif len(families) >= 4:
            return "Оркестр или большой ансамбль"
        elif "клавишные" in families and len(families) == 1:
            return "Фортепианная пьеса"
        else:
            return "Смешанный ансамбль"

    def _analyze_harmony(self, score: Score) -> Dict[str, Any]:
        """Гармонический анализ."""
        chords = []
        key_analysis = {"detected_key": None, "confidence": 0, "modulations": []}

        try:
            for part in score.parts:
                try:
                    flat_part = part.flatten()
                    key_sigs = list(flat_part.getElementsByClass(key.KeySignature))
                    if key_sigs:
                        ks = key_sigs[0]
                        key_analysis["detected_key"] = str(ks)
                        key_analysis["confidence"] = 0.8
                        break
                except:
                    continue

            for part in score.parts:
                try:
                    flat_part = part.flatten()
                    for ch in flat_part.getElementsByClass(chord.Chord):
                        if hasattr(ch, 'pitches') and len(ch.pitches) >= 3:
                            chord_name = ch.pitchedCommonName
                            chords.append({
                                "name": chord_name,
                                "pitches": [p.midi for p in ch.pitches],
                                "duration": float(ch.duration.quarterLength) if hasattr(ch.duration,
                                                                                        'quarterLength') else 0
                            })
                except:
                    continue

            if chords:
                chord_counts = {}
                for ch in chords:
                    name = ch["name"]
                    chord_counts[name] = chord_counts.get(name, 0) + 1

                most_common = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:5]

                harmony_type = "Простая"
                if len(chord_counts) > 10:
                    harmony_type = "Сложная"
                elif any("7" in name for name in chord_counts.keys()):
                    harmony_type = "Джазовая"

                return {
                    "total_chords": len(chords),
                    "unique_chords": len(chord_counts),
                    "most_common_chords": [{"chord": name, "count": count} for name, count in most_common],
                    "harmony_type": harmony_type,
                    "key_signature": key_analysis,
                    "chord_density": round(len(chords) / score.duration.quarterLength, 2) if hasattr(score,
                                                                                                     'duration') else 0
                }
        except Exception as e:
            print(f"   ⚠ Ошибка гармонического анализа: {e}")

        return {
            "total_chords": 0,
            "unique_chords": 0,
            "most_common_chords": [],
            "harmony_type": "Не определено",
            "key_signature": key_analysis,
            "chord_density": 0
        }

    def _analyze_tessitura(self, part) -> Dict[str, Any]:
        """Анализ тесситуры партии."""
        try:
            notes = list(part.flatten().notes)
            if notes:
                pitches = []
                for n in notes:
                    if hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
                        pitches.append(n.pitch.midi)
                if pitches:
                    return {
                        "lowest": min(pitches),
                        "highest": max(pitches),
                        "average": round(statistics.mean(pitches), 1),
                        "range": max(pitches) - min(pitches),
                        "register": self._get_register_description(statistics.mean(pitches))
                    }
        except:
            pass
        return {"lowest": None, "highest": None, "average": None, "range": 0, "register": "Не определен"}

    def _get_register_description(self, avg_pitch: float) -> str:
        """Описание регистра по средней высоте."""
        if avg_pitch < 48:
            return "Очень низкий"
        elif avg_pitch < 60:
            return "Низкий"
        elif avg_pitch < 72:
            return "Средний"
        elif avg_pitch < 84:
            return "Высокий"
        else:
            return "Очень высокий"

    def _analyze_statistics(self, score: Score) -> Dict[str, Any]:
        """Статистический анализ."""
        notes = []
        try:
            flat_score = score.flatten()
            notes = list(flat_score.notes)
        except:
            pass

        if not notes:
            return {"total_notes": 0, "total_rests": 0, "note_range": {"lowest": None, "highest": None},
                    "average_duration": 0, "duration_distribution": {}}

        pitches = []
        durations = []

        for n in notes:
            try:
                if hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
                    pitches.append(n.pitch.midi)
            except:
                pass

            try:
                if hasattr(n, 'duration') and hasattr(n.duration, 'quarterLength'):
                    durations.append(n.duration.quarterLength)
            except:
                pass

        note_range = {"lowest": None, "highest": None}
        if pitches:
            note_range["lowest"] = min(pitches)
            note_range["highest"] = max(pitches)

        average_duration = 0
        if durations:
            average_duration = sum(durations) / len(durations)

        total_rests = 0
        try:
            flat_score = score.flatten()
            total_rests = len(list(flat_score.getElementsByClass('Rest')))
        except:
            pass

        return {
            "total_notes": len(notes),
            "total_rests": total_rests,
            "note_range": note_range,
            "average_duration": round(average_duration, 3),
            "duration_distribution": self._analyze_duration_distribution(notes),
            "rhythmic_complexity": self._calculate_rhythmic_complexity(durations) if durations else 0
        }

    def _analyze_duration_distribution(self, notes) -> Dict[str, float]:
        distribution = {
            "whole_notes": 0,
            "half_notes": 0,
            "quarter_notes": 0,
            "eighth_notes": 0,
            "sixteenth_notes": 0,
            "shorter": 0
        }

        for note in notes:
            try:
                if hasattr(note, 'duration') and hasattr(note.duration, 'quarterLength'):
                    duration = note.duration.quarterLength
                    if duration >= 4:
                        distribution["whole_notes"] += 1
                    elif duration >= 2:
                        distribution["half_notes"] += 1
                    elif duration >= 1:
                        distribution["quarter_notes"] += 1
                    elif duration >= 0.5:
                        distribution["eighth_notes"] += 1
                    elif duration >= 0.25:
                        distribution["sixteenth_notes"] += 1
                    else:
                        distribution["shorter"] += 1
            except:
                continue

        total = len(notes)
        if total > 0:
            for key in distribution:
                distribution[key] = round((distribution[key] / total) * 100, 2)

        return distribution

    def _calculate_rhythmic_complexity(self, durations) -> float:
        """Вычисляет сложность ритма."""
        if not durations:
            return 0

        unique_durations = len(set(round(d, 2) for d in durations))
        total_notes = len(durations)
        short_notes = sum(1 for d in durations if d < 0.25)
        short_percentage = short_notes / total_notes

        complexity = (unique_durations / 10 * 0.4) + (short_percentage * 0.6)
        return round(min(complexity * 10, 10), 2)

    def _analyze_dynamics(self, score: Score) -> Dict[str, Any]:
        """Анализ динамических оттенков."""
        dynamics_list = []
        dynamic_levels = set()

        try:
            for part in score.parts:
                flat_part = part.flatten()
                for dyn in flat_part.getElementsByClass(dynamics.Dynamic):
                    dyn_text = str(dyn)
                    dynamics_list.append(dyn_text)
                    dynamic_levels.add(dyn_text)

                for text in flat_part.getElementsByClass('TextExpression'):
                    text_str = str(text).lower()
                    if any(dyn in text_str for dyn in ['piano', 'forte', 'crescendo', 'diminuendo', 'sforzando']):
                        dynamics_list.append(str(text))
        except:
            pass

        if dynamics_list:
            dynamic_counts = {}
            for dyn in dynamics_list:
                dynamic_counts[dyn] = dynamic_counts.get(dyn, 0) + 1

            return {
                "dynamic_markings": list(dynamic_levels),
                "total_dynamic_marks": len(dynamics_list),
                "dynamic_distribution": dynamic_counts,
                "dynamic_range": self._assess_dynamic_range(dynamic_levels),
                "has_crescendo": any('cresc' in str(d).lower() for d in dynamics_list),
                "has_diminuendo": any('dim' in str(d).lower() for d in dynamics_list)
            }

        return {
            "dynamic_markings": [],
            "total_dynamic_marks": 0,
            "dynamic_distribution": {},
            "dynamic_range": "Не определен",
            "has_crescendo": False,
            "has_diminuendo": False
        }

    def _assess_dynamic_range(self, dynamic_levels) -> str:
        """Оценивает динамический диапазон."""
        levels = list(dynamic_levels)
        has_soft = any(p in str(level).lower() for level in levels for p in ['pp', 'p', 'piano'])
        has_loud = any(f in str(level).lower() for level in levels for f in ['ff', 'f', 'forte'])

        if has_soft and has_loud:
            return "Широкий"
        elif has_soft or has_loud:
            return "Умеренный"
        else:
            return "Ограниченный"

    def _analyze_form_structure(self, score: Score) -> Dict[str, Any]:
        """Анализ структуры формы."""
        sections = []
        repeats = []
        endings = []

        try:
            for elem in score.recurse():
                elem_str = str(elem).lower()
                if 'repeat' in elem_str:
                    repeats.append(str(elem))
                if 'ending' in elem_str or 'volta' in elem_str:
                    endings.append(str(elem))

                if hasattr(elem, 'className') and 'TextExpression' in elem.className:
                    text = str(elem).lower()
                    if any(marker in text for marker in
                           ['verse', 'chorus', 'bridge', 'intro', 'outro', 'solo', 'interlude']):
                        sections.append({
                            "marker": str(elem),
                            "type": self._identify_section_type(text),
                            "position": self._estimate_position(elem, score)
                        })
        except:
            pass

        texture_changes = self._analyze_texture_changes(score)

        return {
            "detected_sections": sections,
            "repeat_structure": repeats,
            "endings": endings,
            "texture_changes": texture_changes,
            "form_type": self._identify_form_type(sections, repeats, texture_changes),
            "section_count": len(sections),
            "has_clear_structure": len(sections) > 0 or len(repeats) > 0
        }

    def _identify_section_type(self, text: str) -> str:
        """Определяет тип секции по текстовому маркеру."""
        text_lower = text.lower()

        if 'verse' in text_lower or 'куплет' in text_lower:
            return "Куплет"
        elif 'chorus' in text_lower or 'припев' in text_lower:
            return "Припев"
        elif 'bridge' in text_lower or 'бридж' in text_lower:
            return "Бридж"
        elif 'intro' in text_lower:
            return "Вступление"
        elif 'outro' in text_lower or 'coda' in text_lower:
            return "Кода"
        elif 'solo' in text_lower:
            return "Соло"
        elif 'interlude' in text_lower:
            return "Интерлюдия"
        else:
            return "Секция"

    def _estimate_position(self, element, score) -> str:
        """Оценивает положение элемента в произведении."""
        try:
            total_measures = 0
            for part in score.parts:
                total_measures = max(total_measures, len(part.getElementsByClass('Measure')))
            return "Не определено"
        except:
            return "Не определено"

    def _analyze_texture_changes(self, score: Score) -> List[Dict[str, Any]]:
        """Анализирует изменения текстуры."""
        changes = []

        try:
            measures = list(score.parts[0].getElementsByClass('Measure')) if score.parts else []
            if len(measures) > 10:
                for i in range(0, len(measures), 10):
                    if i + 10 < len(measures):
                        segment = measures[i:i + 10]
                        texture = self._assess_texture(segment)
                        changes.append({
                            "measure_range": f"{i + 1}-{i + 10}",
                            "texture": texture
                        })
        except:
            pass

        return changes

    def _assess_texture(self, measures) -> str:
        """Оценивает текстуру сегмента."""
        return "Смешанная"

    def _identify_form_type(self, sections, repeats, texture_changes) -> str:
        """Определяет тип формы."""
        if not sections and not repeats:
            return "Сквозная форма"

        section_types = [s["type"] for s in sections]

        if "Куплет" in section_types and "Припев" in section_types:
            if section_types.count("Припев") > 1:
                return "Куплетно-припевная форма"
            else:
                return "Простая куплетная форма"
        elif repeats and len(repeats) >= 2:
            return "Форма с репризами"
        elif len(sections) >= 3:
            return "Сложная секционная форма"
        else:
            return "Простая форма"

    def _estimate_difficulty(self, score) -> Dict[str, Any]:
        """Оценивает сложность произведения."""
        notes = []
        try:
            flat_score = score.flatten()
            notes = list(flat_score.notes)
        except:
            pass

        if not notes:
            return {"level": "Неизвестно", "score": 0, "max_notes_per_chord": 0, "factors": []}

        difficulty_score = 0
        difficulty_factors = []

        pitches = []
        for n in notes:
            try:
                if hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
                    pitches.append(n.pitch.midi)
            except:
                continue

        if pitches:
            range_size = max(pitches) - min(pitches)
            range_score = min(range_size / 20, 5)
            difficulty_score += range_score
            if range_score > 3:
                difficulty_factors.append(f"Широкий диапазон ({range_size} полутонов)")

        max_notes_per_chord = 0
        try:
            flat_score = score.flatten()
            for chord_elem in flat_score.getElementsByClass('Chord'):
                try:
                    if hasattr(chord_elem, 'pitches'):
                        chord_size = len(chord_elem.pitches)
                        max_notes_per_chord = max(max_notes_per_chord, chord_size)
                except:
                    continue
        except:
            pass

        chord_score = min(max_notes_per_chord, 3)
        difficulty_score += chord_score
        if max_notes_per_chord >= 4:
            difficulty_factors.append(f"Сложные аккорды (до {max_notes_per_chord} нот)")

        short_notes = 0
        for n in notes:
            try:
                if hasattr(n, 'duration') and hasattr(n.duration, 'quarterLength'):
                    if n.duration.quarterLength < 0.25:
                        short_notes += 1
            except:
                continue

        if notes:
            short_note_percentage = short_notes / len(notes)
            short_score = short_note_percentage * 10
            difficulty_score += short_score
            if short_note_percentage > 0.3:
                difficulty_factors.append("Много быстрых нот")

        try:
            articulations_count = 0
            for n in notes:
                if hasattr(n, 'articulations') and n.articulations:
                    articulations_count += 1

            if articulations_count > len(notes) * 0.1:
                difficulty_score += 2
                difficulty_factors.append("Сложная артикуляция")
        except:
            pass

        if difficulty_score < 5:
            level = "Начальный"
        elif difficulty_score < 10:
            level = "Средний"
        elif difficulty_score < 15:
            level = "Продвинутый"
        else:
            level = "Экспертный"

        return {
            "level": level,
            "score": round(difficulty_score, 2),
            "max_notes_per_chord": max_notes_per_chord,
            "factors": difficulty_factors
        }

    def _analyze_style(self, score) -> Dict[str, Any]:
        """Анализирует стилистические особенности."""
        style_info = {
            "period_indicators": [],
            "genre_indicators": [],
            "stylistic_features": []
        }

        try:
            flat_score = score.flatten()
            notes = list(flat_score.notes)

            if notes:
                articulations = []
                for note in notes:
                    try:
                        if hasattr(note, 'articulations') and note.articulations:
                            for a in note.articulations:
                                art_str = str(a)
                                articulations.append(art_str)

                                if "Staccato" in art_str:
                                    style_info["period_indicators"].append("Классическая/Романтическая артикуляция")
                                if "Tenuto" in art_str:
                                    style_info["period_indicators"].append("Экспрессивная фразировка")
                                if "Marcato" in art_str:
                                    style_info["period_indicators"].append("Акцентированное исполнение")
                                if "Fermata" in art_str:
                                    style_info["period_indicators"].append("Рубато и агогика")
                    except:
                        continue

                dynamics_info = self._analyze_dynamics(score)
                if dynamics_info["has_crescendo"] and dynamics_info["has_diminuendo"]:
                    style_info["stylistic_features"].append("Динамические контрасты")

                durations = []
                for n in notes:
                    try:
                        if hasattr(n, 'duration') and hasattr(n.duration, 'quarterLength'):
                            durations.append(n.duration.quarterLength)
                    except:
                        continue

                if durations:
                    syncopation = self._detect_syncopation(durations)
                    if syncopation:
                        style_info["genre_indicators"].append("Синкопированный ритм (джаз/поп)")

                    if self._detect_swing(durations):
                        style_info["genre_indicators"].append("Свинговая ритмика (джаз)")

        except Exception as e:
            print(f"   ⚠ Ошибка стилистического анализа: {e}")

        style_info["period_indicators"] = list(set(style_info["period_indicators"]))
        style_info["genre_indicators"] = list(set(style_info["genre_indicators"]))
        style_info["stylistic_features"] = list(set(style_info["stylistic_features"]))

        return style_info

    def _detect_syncopation(self, durations) -> bool:
        """Обнаруживает синкопирование."""
        short_durations = [d for d in durations if d < 0.5]
        return len(short_durations) > len(durations) * 0.4

    def _detect_swing(self, durations) -> bool:
        """Обнаруживает свинговую ритмику."""
        return False

    def _determine_ensemble_type(self, instrumentation_info) -> str:
        """Определяет тип ансамбля."""
        families = instrumentation_info.get("instrument_families", [])
        total_types = instrumentation_info.get("total_instrument_types", 0)

        if "вокальные" in families:
            return "Вокальный ансамбль"
        elif total_types == 1:
            return "Сольное произведение"
        elif total_types <= 4:
            return "Камерный ансамбль"
        elif total_types > 8:
            return "Оркестр"
        else:
            return "Инструментальный ансамбль"

    def _predict_genre(self, score, metadata) -> Dict[str, Any]:
        """Предсказывает жанр произведения."""
        predictions = []
        confidence = 0

        stats = metadata.get("statistical_info", {})
        harmony = metadata.get("harmonic_info", {})
        dynamics_info = metadata.get("dynamic_info", {})

        if stats.get("total_notes", 0) > 1000:
            predictions.append("Оркестровая музыка")
            confidence += 0.3

        if harmony.get("harmony_type") == "Джазовая":
            predictions.append("Джаз")
            confidence += 0.4

        if stats.get("rhythmic_complexity", 0) > 5:
            predictions.append("Современная академическая музыка")
            confidence += 0.2

        if dynamics_info.get("dynamic_range") == "Широкий":
            predictions.append("Романтическая музыка")
            confidence += 0.2

        if not predictions:
            predictions.append("Популярная/Народная музыка")
            confidence = 0.5

        return {
            "predicted_genres": predictions,
            "confidence": min(confidence, 1.0),
            "primary_genre": predictions[0] if predictions else "Не определен"
        }

    def create_metadata_pdf(self, metadata: Dict[str, Any], output_pdf_path: str) -> None:
        """Создает PDF-страницу с метаданными."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import mm
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.lib import colors

            doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                                    topMargin=20 * mm, bottomMargin=20 * mm,
                                    leftMargin=20 * mm, rightMargin=20 * mm)

            story = []
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Title'],
                fontSize=18,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=TA_CENTER
            )

            heading_style = ParagraphStyle(
                'HeadingStyle',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=colors.darkgreen,
                spaceBefore=20,
                spaceAfter=10,
                alignment=TA_LEFT
            )

            subheading_style = ParagraphStyle(
                'SubHeadingStyle',
                parent=styles['Heading3'],
                fontSize=11,
                textColor=colors.darkred,
                spaceBefore=15,
                spaceAfter=5,
                alignment=TA_LEFT
            )

            normal_style = ParagraphStyle(
                'NormalStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.black,
                spaceAfter=5,
                alignment=TA_JUSTIFY
            )

            info_style = ParagraphStyle(
                'InfoStyle',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.darkslategray,
                leftIndent=10,
                spaceAfter=3,
                alignment=TA_LEFT
            )

            title_text = metadata.get('basic_info', {}).get('title', 'Без названия')
            composer = metadata.get('basic_info', {}).get('composer', 'Неизвестный композитор')

            story.append(Paragraph(f"МУЗЫКАЛЬНЫЙ АНАЛИЗ", title_style))
            story.append(Spacer(1, 5 * mm))
            story.append(Paragraph(f"<b>Произведение:</b> {title_text}", heading_style))
            story.append(Paragraph(f"<b>Композитор:</b> {composer}", heading_style))
            story.append(Spacer(1, 10 * mm))

            story.append(Paragraph("1. ОСНОВНАЯ ИНФОРМАЦИЯ", subheading_style))

            basic_info = metadata.get('basic_info', {})
            info_items = [
                ("Опус", basic_info.get('opus_number')),
                ("Номер части", basic_info.get('movement_number')),
                ("Название части", basic_info.get('movement_name')),
                ("Аранжировщик", basic_info.get('arranger')),
                ("Дата создания", basic_info.get('date_created')),
                ("Авторские права", basic_info.get('copyright'))
            ]

            for label, value in info_items:
                if value:
                    story.append(Paragraph(f"<b>{label}:</b> {value}", info_style))

            if not any(value for _, value in info_items):
                story.append(Paragraph("Дополнительная информация отсутствует", info_style))

            story.append(Spacer(1, 5 * mm))

            story.append(Paragraph("2. ИНСТРУМЕНТОВКА", subheading_style))

            instr_info = metadata.get('instrumentation_info', {})
            struct_info = metadata.get('structural_info', {})

            story.append(
                Paragraph(f"<b>Тип ансамбля:</b> {instr_info.get('ensemble_category', 'Не определен')}", info_style))
            story.append(Paragraph(f"<b>Количество партий:</b> {struct_info.get('parts_count', 0)}", info_style))

            families = instr_info.get('instrument_families', [])
            if families:
                story.append(Paragraph(f"<b>Семейства инструментов:</b> {', '.join(families)}", info_style))

            story.append(Spacer(1, 5 * mm))

            parts = struct_info.get('parts', [])
            if parts:
                story.append(Paragraph("<b>Инструменты по партиям:</b>", normal_style))
                for part in parts[:10]:
                    part_name = part.get('part_name', '')
                    instrument = part.get('instrument', '')
                    measures = part.get('measures_count', 0)
                    tessitura = part.get('tessitura', {})
                    register = tessitura.get('register', '')

                    text = f"• {part_name}: {instrument} ({measures} тактов"
                    if register and register != "Не определен":
                        text += f", {register}"
                    text += ")"
                    story.append(Paragraph(text, info_style))
            story.append(Spacer(1, 10 * mm))

            story.append(Paragraph("3. ГАРМОНИЧЕСКИЙ АНАЛИЗ", subheading_style))

            harmony_info = metadata.get('harmonic_info', {})

            story.append(
                Paragraph(f"<b>Тип гармонии:</b> {harmony_info.get('harmony_type', 'Не определено')}", info_style))
            story.append(Paragraph(f"<b>Всего аккордов:</b> {harmony_info.get('total_chords', 0)}", info_style))
            story.append(Paragraph(f"<b>Уникальных аккордов:</b> {harmony_info.get('unique_chords', 0)}", info_style))

            key_sig = harmony_info.get('key_signature', {})
            if key_sig.get('detected_key'):
                story.append(Paragraph(f"<b>Тональность:</b> {key_sig['detected_key']}", info_style))

            common_chords = harmony_info.get('most_common_chords', [])
            if common_chords:
                story.append(Spacer(1, 3 * mm))
                story.append(Paragraph("<b>Наиболее частые аккорды:</b>", normal_style))
                for chord_info in common_chords[:5]:
                    story.append(
                        Paragraph(f"• {chord_info.get('chord', '')} ({chord_info.get('count', 0)} раз)", info_style))

            story.append(Spacer(1, 10 * mm))

            story.append(Paragraph("4. ФОРМА И СТРУКТУРА", subheading_style))

            form_info = metadata.get('form_analysis', {})
            structural_info = metadata.get('structural_info', {})

            story.append(Paragraph(f"<b>Тип формы:</b> {form_info.get('form_type', 'Не определен')}", info_style))
            story.append(Paragraph(f"<b>Всего тактов:</b> {structural_info.get('total_measures', 0)}", info_style))
            story.append(
                Paragraph(f"<b>Размеры:</b> {', '.join(structural_info.get('time_signatures', [])) or 'Не определены'}",
                          info_style))
            story.append(Paragraph(f"<b>Обнаружено секций:</b> {form_info.get('section_count', 0)}", info_style))
            story.append(
                Paragraph(f"<b>Повторения:</b> {'Да' if form_info.get('repeat_structure') else 'Нет'}", info_style))

            story.append(Spacer(1, 10 * mm))

            story.append(Paragraph("5. СТАТИСТИКА И СЛОЖНОСТЬ", subheading_style))

            stats_info = metadata.get('statistical_info', {})
            analysis_info = metadata.get('analysis_info', {})
            difficulty = analysis_info.get('estimated_difficulty', {})

            story.append(Paragraph(f"<b>Всего нот:</b> {stats_info.get('total_notes', 0)}", info_style))
            story.append(Paragraph(f"<b>Всего пауз:</b> {stats_info.get('total_rests', 0)}", info_style))

            note_range = stats_info.get('note_range', {})
            if note_range.get('lowest') and note_range.get('highest'):
                story.append(
                    Paragraph(f"<b>Диапазон нот:</b> MIDI {note_range['lowest']}-{note_range['highest']}", info_style))

            story.append(Paragraph(f"<b>Уровень сложности:</b> {difficulty.get('level', 'Неизвестно')}", info_style))
            story.append(Paragraph(f"<b>Оценка сложности:</b> {difficulty.get('score', 0)}/20", info_style))
            story.append(
                Paragraph(f"<b>Ритмическая сложность:</b> {stats_info.get('rhythmic_complexity', 0)}/10", info_style))

            difficulty_factors = difficulty.get('factors', [])
            if difficulty_factors:
                story.append(Spacer(1, 3 * mm))
                story.append(Paragraph("<b>Факторы сложности:</b>", normal_style))
                for factor in difficulty_factors:
                    story.append(Paragraph(f"• {factor}", info_style))

            story.append(Spacer(1, 10 * mm))

            story.append(Paragraph("6. ЖАНР И СТИЛЬ", subheading_style))

            genre_prediction = analysis_info.get('genre_prediction', {})
            style_indicators = analysis_info.get('style_indicators', {})

            story.append(
                Paragraph(f"<b>Основной жанр:</b> {genre_prediction.get('primary_genre', 'Не определен')}", info_style))
            story.append(
                Paragraph(f"<b>Уверенность предсказания:</b> {genre_prediction.get('confidence', 0) * 100:.0f}%",
                          info_style))

            genre_indicators = style_indicators.get('genre_indicators', [])
            if genre_indicators:
                story.append(Spacer(1, 3 * mm))
                story.append(Paragraph("<b>Жанровые индикаторы:</b>", normal_style))
                for indicator in genre_indicators[:3]:
                    story.append(Paragraph(f"• {indicator}", info_style))

            stylistic_features = style_indicators.get('stylistic_features', [])
            if stylistic_features:
                story.append(Spacer(1, 3 * mm))
                story.append(Paragraph("<b>Стилистические особенности:</b>", normal_style))
                for feature in stylistic_features[:3]:
                    story.append(Paragraph(f"• {feature}", info_style))

            story.append(Spacer(1, 15 * mm))

            story.append(Paragraph("ИНФОРМАЦИЯ О КОНВЕРТАЦИИ", subheading_style))

            conversion_info = metadata.get('conversion_info', {})
            story.append(Paragraph(f"<b>Дата конвертации:</b> {conversion_info.get('conversion_date', 'Неизвестно')}",
                                   info_style))
            story.append(
                Paragraph(f"<b>Версия конвертера:</b> {conversion_info.get('converter_version', 'Неизвестно')}",
                          info_style))
            story.append(Paragraph(f"<b>Версия LilyPond:</b> {conversion_info.get('lilypond_version', 'Неизвестно')}",
                                   info_style))

            doc.build(story)
            print(f"   ✓ PDF с метаданными создан: {Path(output_pdf_path).name}")

        except Exception as e:
            print(f"   ⚠ Ошибка при создании PDF с метаданными: {e}")
            try:
                c = canvas.Canvas(output_pdf_path)
                c.setFont("Helvetica", 14)
                c.drawString(100, 750, "Музыкальный анализ")
                c.setFont("Helvetica", 12)
                c.drawString(100, 730, f"Название: {metadata.get('basic_info', {}).get('title', 'Без названия')}")
                c.drawString(100, 710, f"Композитор: {metadata.get('basic_info', {}).get('composer', 'Неизвестно')}")
                c.save()
            except:
                pass

    def merge_pdfs(self, pdf1_path: str, pdf2_path: str, output_path: str) -> None:
        """Объединяет два PDF файла."""
        try:
            from PyPDF2 import PdfMerger
            merger = PdfMerger()
            merger.append(pdf1_path)
            merger.append(pdf2_path)
            merger.write(output_path)
            merger.close()
        except ImportError:
            print("   Предупреждение: PyPDF2 не установлен, используем альтернативный метод")
            try:
                subprocess.run(['pdftk', pdf1_path, pdf2_path, 'cat', 'output', output_path],
                               check=True, capture_output=True)
            except:
                import shutil
                shutil.copy2(pdf1_path, output_path)
                print("   Не удалось объединить PDF файлы, сохранен только основной PDF")

    # ========== ОСНОВНОЙ МЕТОД КОНВЕРТАЦИИ ==========
    def convert_file(self, input_path: str, output_path: Optional[str] = None,
                     scale_factor: float = 0.8, add_page_numbers: bool = False,
                     include_metadata: bool = True, include_piano_roll: bool = False,
                     **kwargs: Any) -> str:
        start_time = time.time()
        try:
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Входной файл не найден: {input_path}")
            if input_file.suffix.lower() not in ['.xml', '.mxl', '.musicxml']:
                print(f"Предупреждение: Нестандартное расширение файла: {input_file.suffix}")

            if output_path is None:
                output_file = input_file.with_suffix('.pdf')
            else:
                output_file = Path(output_path)
                if output_file.is_dir():
                    output_file = output_file / input_file.with_suffix('.pdf').name

            print(f"\nКонвертация: {input_file.name} -> {output_file.name}")
            print("-" * 50)

            print("1. Чтение MusicXML файла...")
            score = converter.parse(str(input_file))
            print(f"   ✓ Файл загружен: {input_file.name}")

            if score.metadata is None:
                score.metadata = metadata.Metadata()
            if not score.metadata.title:
                score.metadata.title = input_file.stem

            structural_metadata = None
            if include_metadata:
                structural_metadata = self.extract_structural_metadata(score)
                print(f"   ✓ Структурные метаданные извлечены")
                metadata_json_path = output_file.with_suffix('.metadata.json')
                with open(metadata_json_path, 'w', encoding='utf-8') as f:
                    json.dump(structural_metadata, f, indent=2, ensure_ascii=False)
                print(f"   ✓ Метаданные сохранены в JSON: {metadata_json_path.name}")
            print("2. Подготовка LilyPond файла...")
            title = ""
            composer = ""
            try:
                if score.metadata and score.metadata.title:
                    title = score.metadata.title
                else:
                    title = input_file.stem

                if score.metadata and score.metadata.composer:
                    composer = score.metadata.composer
                else:
                    composer = "Неизвестный"
            except:
                title = input_file.stem
                composer = "Неизвестный"

            title = title.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
            composer = composer.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')

            lilypond_settings = f"""
\\version "2.24.0"

\\header {{
    title = "{title}"
    composer = "{composer}"
    tagline = ##f
}}

\\layout {{
    indent = #0
    short-indent = #0
    ragged-last = ##f
    ragged-right = ##f
    ragged-last-bottom = ##f

    \\context {{
        \\Score
        \\override SpacingSpanner.common-shortest-duration = #(ly:make-moment 1/16)
        \\override BarNumber.break-visibility = ##(#f #f #f)
        \\remove "Page_turn_engraver"
    }}

    \\context {{
        \\Staff
        \\remove "Page_turn_engraver"
    }}
}}

\\paper {{
    paper-width = {210 * scale_factor} \\mm
    paper-height = {297 * scale_factor} \\mm
    top-margin = 20 \\mm
    bottom-margin = 20 \\mm
    left-margin = 20 \\mm
    right-margin = 20 \\mm

    print-page-number = ##f
    print-first-page-number = ##f

    oddHeaderMarkup = ##f
    evenHeaderMarkup = ##f
    oddFooterMarkup = ##f
    evenFooterMarkup = ##f
}}
"""
            temp_ly_file = output_file.with_suffix('.temp.ly')

            try:
                score.write('lilypond', fp=str(temp_ly_file))
            except Exception as e:
                print(f"   ⚠ Ошибка при сохранении LilyPond файла: {e}")
                try:
                    with open(temp_ly_file, 'w', encoding='utf-8') as f:
                        f.write(
                            lilypond_settings + "\n\n\\score {\n  \\new Staff {\n    c'4 d' e' f'\n  }\n  \\layout { }\n}")
                    print(f"   ⚠ Создан упрощенный LilyPond файл")
                except:
                    raise RuntimeError(f"Не удалось создать LilyPond файл: {e}")

            with open(temp_ly_file, 'r', encoding='utf-8') as f:
                ly_content = f.read()

            if '\\version' in ly_content:
                lines = ly_content.split('\n')
                new_lines = []
                version_found = False

                for line in lines:
                    new_lines.append(line)
                    if '\\version' in line and not version_found:
                        new_lines.append(lilypond_settings)
                        version_found = True

                if not version_found:
                    new_content = lilypond_settings + '\n' + ly_content
                else:
                    new_content = '\n'.join(new_lines)
            else:
                new_content = lilypond_settings + '\n' + ly_content

            with open(temp_ly_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"   ✓ LilyPond файл подготовлен: {temp_ly_file.name}")
            print(f"   Размер файла: {temp_ly_file.stat().st_size} байт")

            print("3. Компиляция PDF с помощью LilyPond...")
            output_dir = output_file.parent
            output_stem = output_file.stem

            temp_output_dir = output_dir / f"temp_{output_stem}"
            temp_output_dir.mkdir(exist_ok=True)

            cmd = [
                self.lilypond_path,
                '--pdf',
                '--output', str(temp_output_dir),
                str(temp_ly_file)
            ]

            print(f"   Команда LilyPond: {' '.join(cmd)}")

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

                print("   ", end='', flush=True)

                try:
                    stdout, stderr = process.communicate(timeout=60)

                    if process.returncode != 0:
                        print(f"\n   ✗ Ошибка компиляции LilyPond (код {process.returncode})")
                        if stderr:
                            print(f"   Ошибки LilyPond:\n{stderr[:1000]}")
                        if stdout:
                            print(f"   Вывод LilyPond:\n{stdout[:1000]}")

                        pdf_files = list(temp_output_dir.glob("*.pdf"))
                        if pdf_files:
                            print(f"   Найдены PDF файлы: {[f.name for f in pdf_files]}")
                            pdf_files[0].rename(output_file)
                            print(f"   PDF найден и переименован: {pdf_files[0].name} -> {output_file.name}")
                            actual_pdf = output_file
                        else:
                            raise RuntimeError(f"LilyPond не создал PDF файл")
                    else:
                        print("\n   ✓ PDF успешно скомпилирован")

                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    raise RuntimeError(f"Таймаут при компиляции LilyPond (больше 60 секунд)")

            except Exception as e:
                if temp_ly_file.exists():
                    temp_ly_file.unlink()
                if temp_output_dir.exists():
                    pdf_files = list(output_dir.glob(f"*.pdf"))
                    if pdf_files:
                        for pdf in pdf_files:
                            if output_stem in pdf.name:
                                pdf.rename(output_file)
                                print(f"   PDF найден и переименован: {pdf.name} -> {output_file.name}")
                                actual_pdf = output_file
                                break
                        else:
                            raise RuntimeError(f"LilyPond ошибка: {str(e)}")
                    else:
                        raise RuntimeError(f"LilyPond ошибка: {str(e)}")
                else:
                    raise RuntimeError(f"LilyPond ошибка: {str(e)}")

            if temp_ly_file.exists():
                temp_ly_file.unlink()
                print(f"   ✓ Временный файл удален: {temp_ly_file.name}")
            actual_pdf = None

            if temp_output_dir.exists():
                pdf_files = list(temp_output_dir.glob("*.pdf"))
                if pdf_files:
                    pdf_file = pdf_files[0]
                    pdf_file.rename(output_file)
                    actual_pdf = output_file
                    print(f"   ✓ PDF найден во временной директории: {pdf_file.name}")
                else:
                    all_files = list(temp_output_dir.glob("*"))
                    print(f"   Файлы во временной директории: {[f.name for f in all_files]}")

            if not actual_pdf or not actual_pdf.exists():
                pdf_files = list(output_dir.glob(f"{output_stem}*.pdf"))
                if pdf_files:
                    pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    actual_pdf = pdf_files[0]
                    if actual_pdf != output_file:
                        actual_pdf.rename(output_file)
                        actual_pdf = output_file
                    print(f"   ✓ PDF найден в основной директории: {actual_pdf.name}")

            if not actual_pdf or not actual_pdf.exists():
                pdf_files = list(output_dir.glob("*.pdf"))
                if pdf_files:
                    for pdf in pdf_files:
                        if output_stem.lower() in pdf.name.lower():
                            pdf.rename(output_file)
                            actual_pdf = output_file
                            print(f"   ✓ PDF найден по части имени: {pdf.name}")
                            break
                    else:
                        pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        actual_pdf = pdf_files[0]
                        new_name = output_dir / f"{output_stem}.pdf"
                        actual_pdf.rename(new_name)
                        actual_pdf = new_name
                        print(f"   ✓ PDF найден и переименован: {pdf_files[0].name} -> {actual_pdf.name}")

            if not actual_pdf or not actual_pdf.exists():
                raise FileNotFoundError(f"PDF файл не был создан. Проверьте вывод LilyPond выше.")

            for ext in ['.log', '.tex', '.count', '.eps', '.png', '.svg']:
                temp_file = output_dir / f"{output_stem}{ext}"
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass

            if temp_output_dir.exists():
                try:
                    for file in temp_output_dir.glob("*"):
                        try:
                            file.unlink()
                        except:
                            pass
                    temp_output_dir.rmdir()
                except:
                    pass

            # Добавление метаданных
            if include_metadata and structural_metadata and actual_pdf:
                print("4. Добавление структурных метаданных в PDF...")
                try:
                    metadata_pdf_path = output_dir / f"{output_stem}_metadata.pdf"
                    self.create_metadata_pdf(structural_metadata, str(metadata_pdf_path))
                    print(f"   ✓ PDF с метаданными создан: {metadata_pdf_path.name}")

                    final_pdf_path = output_dir / f"{output_stem}_with_metadata.pdf"
                    self.merge_pdfs(str(actual_pdf), str(metadata_pdf_path), str(final_pdf_path))

                    actual_pdf.unlink()
                    final_pdf_path.rename(actual_pdf)
                    metadata_pdf_path.unlink()

                    print(f"   ✓ Метаданные добавлены в PDF")
                except Exception as e:
                    print(f"   ⚠ Ошибка при добавлении метаданных: {e}")
                    print(f"   PDF сохранен без метаданных")

            # Добавление пиано-ролла
            if include_piano_roll and actual_pdf:
                print("5. Добавление пиано-ролла в PDF...")
                try:
                    piano_roll_path = output_dir / f"{output_stem}_pianoroll.pdf"
                    self.generate_piano_roll_pdf(score, str(piano_roll_path))

                    temp_merged = output_dir / f"{output_stem}_with_pianoroll.pdf"
                    self.merge_pdfs(str(actual_pdf), str(piano_roll_path), str(temp_merged))

                    actual_pdf.unlink()
                    temp_merged.rename(actual_pdf)
                    piano_roll_path.unlink()

                    print(f"   ✓ Пиано-ролл добавлен в PDF")
                except Exception as e:
                    print(f"   ⚠ Ошибка при добавлении пиано-ролла: {e}")

            elapsed_time = time.time() - start_time
            file_size = actual_pdf.stat().st_size / 1024

            print("-" * 50)
            print(f"✓ Конвертация завершена успешно!")
            print(f"  Время выполнения: {elapsed_time:.1f} сек")
            print(f"  Размер PDF: {file_size:.1f} КБ")
            print(f"  PDF файл: {actual_pdf}")

            if include_metadata:
                print(f"  JSON с метаданными: {metadata_json_path}")

            return str(actual_pdf)

        except Exception as e:
            print(f"\n✗ Ошибка при конвертации:")
            print(f"  {type(e).__name__}: {str(e)}")

            print(f"\nДИАГНОСТИКА:")
            print(f"  Входной файл: {input_path}")
            print(f"  Выходной файл: {output_path if output_path else 'Не указан'}")
            print(f"  LilyPond путь: {self.lilypond_path}")
            print(f"  LilyPond доступен: {os.path.exists(self.lilypond_path) if self.lilypond_path else 'Нет'}")

            try:
                if 'temp_ly_file' in locals() and temp_ly_file.exists():
                    print(f"  LilyPond файл создан: Да ({temp_ly_file.stat().st_size} байт)")
                    with open(temp_ly_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"  Первые 500 символов LilyPond файла:")
                        print(f"  {content[:500]}...")
                else:
                    print(f"  LilyPond файл создан: Нет")
            except:
                pass

            raise

    def batch_convert(self, input_files: List[str], output_dir: Optional[str] = None,
                      include_metadata: bool = True, include_piano_roll: bool = False) -> List[str]:
        results = []
        total_files = len(input_files)
        print(f"\nНачало пакетной конвертации {total_files} файлов")
        print("=" * 60)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        for i, input_file in enumerate(input_files, 1):
            try:
                print(f"\n[{i}/{total_files}] Конвертация файла: {Path(input_file).name}")
                if output_dir:
                    output_file = output_path / Path(input_file).with_suffix('.pdf').name
                else:
                    output_file = None
                result = self.convert_file(input_file, output_file, include_metadata=include_metadata,
                                           include_piano_roll=include_piano_roll)
                results.append(result)
                print(f"   ✓ Успешно")
            except Exception as e:
                print(f"   ✗ Ошибка: {str(e)[:100]}")
                results.append(None)
        successful = sum(1 for r in results if r is not None)
        failed = total_files - successful
        print("\n" + "=" * 60)
        print(f"Пакетная конвертация завершена:")
        print(f"  Успешно: {successful}")
        print(f"  Не удалось: {failed}")
        return results

    def get_lilypond_info(self) -> Dict[str, str]:
        info = {
            "path": self.lilypond_path,
            "exists": str(os.path.exists(self.lilypond_path)) if self.lilypond_path else "False",
            "platform": sys.platform,
            "default_path": self.DEFAULT_LILYPOND_PATH,
        }
        try:
            result = subprocess.run(
                [self.lilypond_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout:
                info["version"] = result.stdout.split('\n')[0].strip()
        except:
            info["version"] = "неизвестна"
        return info


def get_file_from_user() -> str:
    """Получает путь к файлу от пользователя через консоль."""
    print("\n" + "=" * 60)
    print("ВВОД ФАЙЛА ДЛЯ КОНВЕРТАЦИИ")
    print("=" * 60)

    while True:
        file_path = input("\nВведите путь к MusicXML файлу (.xml, .mxl, .musicxml): ").strip()

        if not file_path:
            print("Путь не может быть пустым. Попробуйте снова.")
            continue

        # Убираем кавычки, если пользователь их использовал
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        elif file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]

        file_path = Path(file_path)

        # Проверяем существование файла
        if not file_path.exists():
            print(f"Файл не найден: {file_path}")
            print("Убедитесь, что путь указан правильно.")
            continue

        # Проверяем расширение
        if file_path.suffix.lower() not in ['.xml', '.mxl', '.musicxml']:
            print(f"Предупреждение: Файл имеет нестандартное расширение: {file_path.suffix}")
            confirm = input("Продолжить? (y/n): ").lower().strip()
            if confirm != 'y':
                continue

        return str(file_path)


def get_output_path_suggestion(input_path: str) -> str:
    """Предлагает путь для сохранения PDF файла."""
    input_file = Path(input_path)
    default_output = input_file.with_suffix('.pdf')

    print(f"\nПредлагаемый путь для сохранения: {default_output}")

    while True:
        choice = input("Использовать предложенный путь? (y/n): ").lower().strip()

        if choice == 'y':
            return str(default_output)
        elif choice == 'n':
            new_path = input("Введите новый путь для сохранения PDF: ").strip()

            if not new_path:
                print("Путь не может быть пустым.")
                continue

            # Убираем кавычки
            if new_path.startswith('"') and new_path.endswith('"'):
                new_path = new_path[1:-1]
            elif new_path.startswith("'") and new_path.endswith("'"):
                new_path = new_path[1:-1]

            new_path = Path(new_path)

            # Если указана директория, создаём там файл
            if new_path.is_dir():
                new_path = new_path / default_output.name

            # Проверяем расширение
            if new_path.suffix.lower() != '.pdf':
                new_path = new_path.with_suffix('.pdf')

            # Проверяем, существует ли файл
            if new_path.exists():
                overwrite = input(f"Файл {new_path} уже существует. Перезаписать? (y/n): ").lower().strip()
                if overwrite != 'y':
                    print("Выберите другой путь.")
                    continue

            return str(new_path)
        else:
            print("Пожалуйста, введите 'y' или 'n'.")


def get_batch_files_from_user() -> List[str]:
    """Получает несколько файлов для пакетной конвертации."""
    print("\n" + "=" * 60)
    print("ПАКЕТНАЯ КОНВЕРТАЦИЯ")
    print("=" * 60)

    files = []

    while True:
        print(f"\nТекущий список файлов ({len(files)}):")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {Path(f).name}")

        print("\nВыберите действие:")
        print("  1. Добавить файл")
        print("  2. Удалить файл")
        print("  3. Начать конвертацию")
        print("  4. Отмена")

        choice = input("\nВаш выбор (1-4): ").strip()

        if choice == '1':  # Добавить файл
            file_path = input("Введите путь к MusicXML файлу: ").strip()

            if not file_path:
                print("Путь не может быть пустым.")
                continue

            # Убираем кавычки
            if file_path.startswith('"') and file_path.endswith('"'):
                file_path = file_path[1:-1]
            elif file_path.startswith("'") and file_path.endswith("'"):
                file_path = file_path[1:-1]

            file_path = Path(file_path)

            if not file_path.exists():
                print(f"Файл не найден: {file_path}")
                continue

            if str(file_path) in files:
                print("Этот файл уже добавлен.")
                continue

            files.append(str(file_path))
            print(f"✓ Файл добавлен: {file_path.name}")

        elif choice == '2':  # Удалить файл
            if not files:
                print("Список файлов пуст.")
                continue

            try:
                index = int(input(f"Введите номер файла для удаления (1-{len(files)}): "))
                if 1 <= index <= len(files):
                    removed = files.pop(index - 1)
                    print(f"✓ Файл удален: {Path(removed).name}")
                else:
                    print("Неверный номер.")
            except ValueError:
                print("Пожалуйста, введите число.")

        elif choice == '3':  # Начать конвертацию
            if not files:
                print("Список файлов пуст. Добавьте файлы сначала.")
                continue

            output_dir = input("Введите директорию для сохранения PDF (Enter для текущей): ").strip()
            if output_dir:
                output_dir = Path(output_dir)
                if not output_dir.exists():
                    create = input(f"Директория не существует. Создать? (y/n): ").lower()
                    if create == 'y':
                        output_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        print("Конвертация отменена.")
                        return []

            return files

        elif choice == '4':  # Отмена
            print("Пакетная конвертация отменена.")
            return []

        else:
            print("Неверный выбор. Пожалуйста, выберите 1-4.")


def main_menu():
    """Главное меню программы."""
    print("\n" + "=" * 60)
    print("MUSICXML В PDF КОНВЕРТЕР СО СТРУКТУРНЫМИ МЕТАДАННЫМИ И ПИАНО-РОЛЛОМ")
    print("=" * 60)
    print("Программа для конвертации MusicXML файлов в PDF")
    print("Использует music21 и LilyPond для рендеринга")

    try:
        # Инициализируем конвертер
        print("\nИнициализация конвертера...")
        converter = MusicXMLtoPDFConverter()

        # Показываем информацию о LilyPond
        info = converter.get_lilypond_info()
        print(f"✓ LilyPond версия: {info.get('version', 'неизвестна')}")

        while True:
            print("\n" + "-" * 40)
            print("ГЛАВНОЕ МЕНЮ")
            print("-" * 40)
            print("1. Конвертировать один файл")
            print("2. Пакетная конвертация нескольких файлов")
            print("3. Показать информацию о LilyPond")
            print("4. Протестировать LilyPond")
            print("5. Выход")

            choice = input("\nВыберите действие (1-5): ").strip()

            if choice == '1':  # Конвертировать один файл
                try:
                    input_file = get_file_from_user()
                    output_file = get_output_path_suggestion(input_file)

                    # Дополнительные настройки
                    print("\nДополнительные настройки:")
                    scale = input(f"Масштаб (0.5-1.5, по умолчанию 0.8): ").strip()
                    scale_factor = float(scale) if scale else 0.8

                    metadata_choice = input(
                        "Добавить структурные метаданные в PDF? (y/n, по умолчанию y): ").strip().lower()
                    include_metadata = metadata_choice != 'n'

                    piano_roll_choice = input(
                        "Добавить пиано-ролл (интервальную спектрограмму)? (y/n, по умолчанию n): ").strip().lower()
                    include_piano_roll = piano_roll_choice == 'y'

                    # Выполняем конвертацию
                    print("\nНачало конвертации...")
                    result = converter.convert_file(
                        input_file,
                        output_file,
                        scale_factor=scale_factor,
                        include_metadata=include_metadata,
                        include_piano_roll=include_piano_roll
                    )

                    print(f"\n✓ Конвертация завершена!")
                    print(f"  Результат: {result}")

                    # Предлагаем открыть файл
                    open_file = input("\nОткрыть PDF файл? (y/n): ").strip().lower()
                    if open_file == 'y':
                        if sys.platform == "win32":
                            os.startfile(result)
                        elif sys.platform == "darwin":
                            subprocess.run(['open', result])
                        else:
                            subprocess.run(['xdg-open', result])

                except Exception as e:
                    print(f"\n✗ Ошибка: {str(e)}")
                    print("Попробуйте снова.")
            elif choice == '2':  # Пакетная конвертация
                try:
                    files = get_batch_files_from_user()
                    if files:
                        output_dir = input("\nВведите директорию для сохранения (Enter для текущей): ").strip()
                        output_dir = output_dir if output_dir else None

                        metadata_choice = input(
                            "Добавить структурные метаданные в PDF? (y/n, по умолчанию y): ").strip().lower()
                        include_metadata = metadata_choice != 'n'

                        piano_roll_choice = input(
                            "Добавить пиано-ролл для каждого файла? (y/n, по умолчанию n): ").strip().lower()
                        include_piano_roll = piano_roll_choice == 'y'

                        print("\nНачало пакетной конвертации...")
                        results = converter.batch_convert(
                            files,
                            output_dir,
                            include_metadata=include_metadata,
                            include_piano_roll=include_piano_roll
                        )

                        print("\nПакетная конвертация завершена!")

                        # Показываем статистику
                        successful = sum(1 for r in results if r is not None)
                        failed = len(files) - successful
                        print(f"  Успешно: {successful}")
                        print(f"  Не удалось: {failed}")

                        if failed > 0:
                            print("\nФайлы с ошибками:")
                            for i, (file, result) in enumerate(zip(files, results)):
                                if result is None:
                                    print(f"  • {Path(file).name}")

                except Exception as e:
                    print(f"\n✗ Ошибка: {str(e)}")
                    print("Попробуйте снова.")

            elif choice == '3':  # Информация о LilyPond
                print("\n" + "-" * 40)
                print("ИНФОРМАЦИЯ О LILYPOND")
                print("-" * 40)
                info = converter.get_lilypond_info()
                for key, value in info.items():
                    print(f"  {key}: {value}")

                # Дополнительная информация о системе
                print("\n  Системная информация:")
                print(f"  Платформа: {sys.platform}")
                print(f"  Python версия: {sys.version.split()[0]}")
                print(
                    f"  Music21 версия: {converter.__version__ if hasattr(converter, '__version__') else 'неизвестно'}")

            elif choice == '4':  # Тестирование LilyPond
                print("\n" + "-" * 40)
                print("ТЕСТИРОВАНИЕ LILYPOND")
                print("-" * 40)

                # Создаем простой тестовый файл
                test_ly = Path("test_lilypond.ly")
                test_pdf = Path("test_lilypond.pdf")

                try:
                    with open(test_ly, 'w', encoding='utf-8') as f:
                        f.write('''
\\version "2.24.0"
\\header { title = "LilyPond Test" }
\\score { \\new Staff { c' d' e' f' g' a' b' c'' } }
                        ''')

                    result = subprocess.run(
                        [converter.lilypond_path, '--pdf', str(test_ly)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode == 0 and test_pdf.exists():
                        print("\n✓ LilyPond работает корректно")
                        print("  Тестовый PDF создан успешно")

                        # Удаляем тестовые файлы
                        test_ly.unlink()
                        test_pdf.unlink()
                    else:
                        print("\n✗ Проблемы с LilyPond. Проверьте установку.")
                        if result.stderr:
                            print(f"  Ошибка: {result.stderr[:200]}")

                except Exception as e:
                    print(f"\n✗ Ошибка при тестировании: {e}")
                    print("\nРекомендации:")
                    print("1. Убедитесь, что LilyPond установлен")
                    print("2. Проверьте путь: C:\\Program Files\\LilyPond\\bin\\lilypond.exe")
                    print("3. Попробуйте переустановить LilyPond")
                    print("4. Проверьте переменные окружения PATH")

                # Очистка
                if test_ly.exists():
                    test_ly.unlink()
                if test_pdf.exists():
                    test_pdf.unlink()

            elif choice == '5':  # Выход
                print("\nСпасибо за использование программы!")
                print("До свидания!")
                break

            else:
                print("Неверный выбор. Пожалуйста, выберите 1-5.")

    except FileNotFoundError as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("\nДля решения проблемы:")
        print("1. Установите LilyPond с https://lilypond.org/download.html")
        print("2. Убедитесь, что установка в папку: C:\\Program Files\\LilyPond")
        print("3. Или укажите путь вручную при запуске:")
        print("   python lilypond.py --lilypond 'ваш\\путь\\lilypond.exe'")
        print("\nАльтернативные пути:")
        converter = MusicXMLtoPDFConverter()  # Временный объект для получения альтернативных путей
        alt_paths = converter._get_alternative_lilypond_paths()
        for path in alt_paths[:5]:
            print(f"   • {path}")

    except Exception as e:
        print(f"\n✗ НЕИЗВЕСТНАЯ ОШИБКА: {type(e).__name__}: {e}")
        print("\nДетальная информация об ошибке:")
        traceback.print_exc()

        # Сохраняем лог ошибки
        error_log = f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"Ошибка: {type(e).__name__}: {e}\n")
            f.write(f"Время: {datetime.now()}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"Платформа: {sys.platform}\n")
            f.write("-" * 50 + "\n")
            traceback.print_exc(file=f)
        print(f"\nЛог ошибки сохранен в: {error_log}")


def main():
    """Точка входа в программу."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Конвертер MusicXML в PDF со структурными метаданными и пиано-роллом',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python lilypond.py --input "file.xml" --output "file.pdf"
  python lilypond.py --input "file.xml" --piano-roll --no-metadata
  python lilypond.py --batch --input "scores/" --output "pdfs/"
  python lilypond.py --batch --input "filelist.txt" --piano-roll
        """
    )

    parser.add_argument('--lilypond', type=str,
                        help='Путь к LilyPond (по умолчанию: C:\\Program Files\\LilyPond\\bin\\lilypond.exe)')
    parser.add_argument('--input', type=str, help='Путь к входному файлу или директории (для пакетного режима)')
    parser.add_argument('--output', type=str, help='Путь к выходному файлу или директории')
    parser.add_argument('--batch', action='store_true', help='Пакетный режим (конвертировать несколько файлов)')
    parser.add_argument('--scale', type=float, default=0.8, help='Масштаб страницы (0.5-1.5, по умолчанию 0.8)')
    parser.add_argument('--no-metadata', action='store_true', help='Не добавлять метаданные в PDF')
    parser.add_argument('--piano-roll', action='store_true',
                        help='Добавить пиано-ролл (интервальную спектрограмму) в PDF')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')

    args = parser.parse_args()

    # Если указаны аргументы командной строки, используем их
    if args.input:
        try:
            if args.verbose:
                print("Запуск в режиме командной строки...")
                print(f"Аргументы: {args}")

            if args.batch:
                # Пакетный режим
                input_files = []

                if os.path.isdir(args.input):
                    # Если указана директория, ищем все MusicXML файлы
                    input_path = Path(args.input)
                    for ext in ['*.xml', '*.mxl', '*.musicxml']:
                        input_files.extend(str(p) for p in input_path.glob(ext))

                    if args.verbose:
                        print(f"Найдено {len(input_files)} файлов в директории {args.input}")

                else:
                    # Если указан файл со списком
                    if args.input.endswith('.txt'):
                        try:
                            with open(args.input, 'r', encoding='utf-8') as f:
                                input_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                            if args.verbose:
                                print(f"Загружено {len(input_files)} файлов из списка {args.input}")
                        except Exception as e:
                            print(f"Ошибка при чтении файла списка: {e}")
                            sys.exit(1)
                    else:
                        input_files = [args.input]

                if not input_files:
                    print("Не найдено файлов для конвертации")
                    sys.exit(1)

                # Создаем конвертер
                converter = MusicXMLtoPDFConverter(args.lilypond)

                # Выполняем пакетную конвертацию
                results = converter.batch_convert(
                    input_files,
                    args.output,
                    include_metadata=not args.no_metadata,
                    include_piano_roll=args.piano_roll
                )

                # Выводим статистику
                successful = sum(1 for r in results if r is not None)
                print(f"\nПакетная конвертация завершена. Успешно: {successful} из {len(input_files)} файлов")

            else:
                # Одиночный файл
                converter = MusicXMLtoPDFConverter(args.lilypond)

                result = converter.convert_file(
                    args.input,
                    args.output,
                    scale_factor=args.scale,
                    include_metadata=not args.no_metadata,
                    include_piano_roll=args.piano_roll
                )

                print(f"Конвертация завершена: {result}")

        except Exception as e:
            print(f"Ошибка: {e}")
            if args.verbose:
                traceback.print_exc()
            sys.exit(1)
    else:
        # Запускаем интерактивное меню
        main_menu()


if __name__ == "__main__":
    main()