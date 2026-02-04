import hashlib
from typing import List, Dict, Any, Optional, Tuple
from music21 import note, interval, pitch, chord, dynamics
from music21.stream.base import Score
import statistics


class NoteAnalyzer:
    """Класс для анализа музыкальных нот и их характеристик."""

    @staticmethod
    def get_note_range_overlap(
            score: Score,
            start_measure: int = 1,
            end_measure: Optional[int] = None,
            part_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Возвращает интервал нот в указанном диапазоне с нахлестом.

        Args:
            score: Music21 Score объект
            start_measure: начальный такт (1-based)
            end_measure: конечный такт (None = до конца)
            part_index: индекс партии (None = все партии)

        Returns:
            Список словарей с характеристиками нот и их токенами
        """
        notes_info = []

        try:
            # Определяем, какие партии анализировать
            if part_index is not None:
                parts_to_analyze = [score.parts[part_index]]
            else:
                parts_to_analyze = score.parts

            for part_idx, part in enumerate(parts_to_analyze):
                part_name = getattr(part, 'partName', f'Part {part_idx + 1}')

                # Получаем такты
                measures = list(part.getElementsByClass('Measure'))

                if not measures:
                    continue

                # Определяем конечный такт
                if end_measure is None:
                    actual_end_measure = len(measures)
                else:
                    actual_end_measure = min(end_measure, len(measures))

                # Собираем ноты из указанного диапазона
                for measure_idx in range(max(0, start_measure - 1), actual_end_measure):
                    measure = measures[measure_idx]
                    measure_num = measure_idx + 1

                    # Получаем все ноты в такте
                    measure_notes = list(measure.notesAndRests)

                    for note_idx, n in enumerate(measure_notes):
                        try:
                            note_info = {
                                'part_index': part_idx,
                                'part_name': part_name,
                                'measure': measure_num,
                                'position_in_measure': note_idx,
                                'note_object': n,
                                'token': NoteAnalyzer._create_note_token(n, part_idx, measure_num, note_idx),
                                'characteristics': NoteAnalyzer._get_note_characteristics(n),
                                'overlap_info': NoteAnalyzer._get_overlap_info(n, measure_notes, note_idx)
                            }
                            notes_info.append(note_info)
                        except Exception as e:
                            print(f"Ошибка обработки ноты: {e}")
                            continue

            # Добавляем информацию о связях между нотами
            NoteAnalyzer._add_note_relations(notes_info)

        except Exception as e:
            print(f"Ошибка при получении диапазона нот: {e}")

        return notes_info

    @staticmethod
    def get_note_intervals(notes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Возвращает интервалы между последовательными нотами.

        Args:
            notes_info: Список информации о нотах

        Returns:
            Список интервалов между нотами
        """
        intervals = []

        try:
            # Группируем ноты по партиям
            parts_dict = {}
            for note_info in notes_info:
                part_idx = note_info.get('part_index')
                if part_idx not in parts_dict:
                    parts_dict[part_idx] = []
                parts_dict[part_idx].append(note_info)

            # Для каждой партии вычисляем интервалы
            for part_idx, part_notes in parts_dict.items():
                # Сортируем ноты по такту и позиции
                part_notes.sort(key=lambda x: (x['measure'], x['position_in_measure']))

                for i in range(1, len(part_notes)):
                    prev_note = part_notes[i - 1]['note_object']
                    curr_note = part_notes[i]['note_object']

                    if (hasattr(prev_note, 'pitch') and hasattr(curr_note, 'pitch') and
                            not isinstance(prev_note, note.Rest) and
                            not isinstance(curr_note, note.Rest)):
                        interval_info = NoteAnalyzer._get_interval_between_notes(prev_note, curr_note)

                        intervals.append({
                            'from_note': NoteAnalyzer._create_note_token(
                                prev_note,
                                part_idx,
                                part_notes[i - 1]['measure'],
                                part_notes[i - 1]['position_in_measure']
                            ),
                            'to_note': NoteAnalyzer._create_note_token(
                                curr_note,
                                part_idx,
                                part_notes[i]['measure'],
                                part_notes[i]['position_in_measure']
                            ),
                            'interval': interval_info,
                            'part_index': part_idx,
                            'from_measure': part_notes[i - 1]['measure'],
                            'to_measure': part_notes[i]['measure'],
                            'time_distance': NoteAnalyzer._calculate_time_gap(prev_note, curr_note)
                        })

        except Exception as e:
            print(f"Ошибка при вычислении интервалов: {e}")

        return intervals

    @staticmethod
    def _create_note_token(
            note_obj: note.Note,
            part_idx: int,
            measure_num: int,
            position: int
    ) -> str:
        """
        Создает уникальный токен для ноты.

        Формат: P{part}M{measure}N{position}_{pitch}_{duration}_{articulations}
        """
        try:
            # Базовые части токена
            base_token = f"P{part_idx}M{measure_num}N{position}"

            # Если это пауза
            if isinstance(note_obj, note.Rest):
                if hasattr(note_obj, 'duration'):
                    duration = note_obj.duration.quarterLength
                    return f"{base_token}_REST_D{duration}"
                return f"{base_token}_REST"

            # Если это аккорд
            if isinstance(note_obj, chord.Chord):
                pitches = [str(p) for p in note_obj.pitches]
                pitches_str = "_".join(pitches)
                duration = note_obj.duration.quarterLength
                return f"{base_token}_CHORD_{pitches_str}_D{duration}"

            # Если это нота
            if hasattr(note_obj, 'pitch'):
                pitch_str = str(note_obj.pitch)
            else:
                pitch_str = 'unknown'

            if hasattr(note_obj, 'duration'):
                duration = note_obj.duration.quarterLength
                duration_str = f"D{duration:.3f}"
            else:
                duration_str = "D0"

            # Артикуляция
            articulations = []
            if hasattr(note_obj, 'articulations'):
                for art in note_obj.articulations:
                    art_name = art.__class__.__name__
                    if art_name not in articulations:
                        articulations.append(art_name)

            articulation_str = "_".join(articulations) if articulations else "no_articulation"

            # Создаем токен
            token = f"{base_token}_{pitch_str}_{duration_str}_{articulation_str}"

            # Хешируем для компактности
            token_hash = hashlib.md5(token.encode()).hexdigest()[:6]

            return f"NT_{token_hash}"

        except Exception as e:
            return f"ERR_{hash(str(e)) % 10000}"

    @staticmethod
    def _get_note_characteristics(note_obj: note.Note) -> Dict[str, Any]:
        """Возвращает характеристики ноты."""
        characteristics = {
            'basic': {},
            'pitch_info': {},
            'rhythmic_info': {},
            'performance_info': {},
            'contextual_info': {}
        }

        try:
            # Базовые характеристики
            characteristics['basic'] = {
                'type': note_obj.className,
                'is_rest': isinstance(note_obj, note.Rest),
                'is_chord': isinstance(note_obj, chord.Chord),
                'is_note': isinstance(note_obj, note.Note)
            }

            # Если это нота (не пауза и не аккорд)
            if isinstance(note_obj, note.Note) and hasattr(note_obj, 'pitch'):
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
                    'spelling': NoteAnalyzer._get_pitch_spelling(p)
                }

            # Ритмическая информация (для всех типов)
            if hasattr(note_obj, 'duration'):
                d = note_obj.duration
                characteristics['rhythmic_info'] = {
                    'quarter_length': d.quarterLength,
                    'type': d.type,
                    'dots': d.dots,
                    'tuplets': [str(t) for t in d.tuplets] if hasattr(d, 'tuplets') else [],
                    'is_grace': d.isGrace,
                    'is_incomplete': d.incompleteFill,
                    'duration_components': NoteAnalyzer._get_duration_components(d)
                }

            # Исполнительская информация
            if not isinstance(note_obj, note.Rest):
                characteristics['performance_info'] = {
                    'articulations': [str(a) for a in note_obj.articulations]
                    if hasattr(note_obj, 'articulations') else [],
                    'expressions': [str(e) for e in note_obj.expressions]
                    if hasattr(note_obj, 'expressions') else [],
                    'lyrics': [str(l.text) for l in note_obj.lyrics]
                    if hasattr(note_obj, 'lyrics') else [],
                    'tie': str(note_obj.tie) if hasattr(note_obj, 'tie') else None,
                    'stem_direction': note_obj.stemDirection if hasattr(note_obj, 'stemDirection') else None
                }

            # Громкость
            if hasattr(note_obj, 'volume'):
                characteristics['performance_info']['volume'] = {
                    'velocity': note_obj.volume.velocity,
                    'velocity_is_relative': note_obj.volume.velocityIsRelative
                }

        except Exception as e:
            characteristics['error'] = str(e)

        return characteristics

    @staticmethod
    def _get_overlap_info(
            current_note: note.Note,
            measure_notes: List,
            position: int
    ) -> Dict[str, Any]:
        """
        Анализирует нахлест ноты с соседними нотами.
        """
        overlap_info = {
            'previous_overlap': None,
            'next_overlap': None,
            'simultaneous_notes': []
        }

        try:
            # Проверяем перекрытие с предыдущей нотой
            if position > 0:
                prev_note = measure_notes[position - 1]
                if NoteAnalyzer._notes_overlap(prev_note, current_note):
                    overlap_info['previous_overlap'] = {
                        'note': str(prev_note),
                        'overlap_type': NoteAnalyzer._determine_overlap_type(prev_note, current_note),
                        'overlap_ratio': NoteAnalyzer._calculate_overlap_ratio(prev_note, current_note)
                    }

            # Проверяем перекрытие со следующей нотой
            if position < len(measure_notes) - 1:
                next_note = measure_notes[position + 1]
                if NoteAnalyzer._notes_overlap(current_note, next_note):
                    overlap_info['next_overlap'] = {
                        'note': str(next_note),
                        'overlap_type': NoteAnalyzer._determine_overlap_type(current_note, next_note),
                        'overlap_ratio': NoteAnalyzer._calculate_overlap_ratio(current_note, next_note)
                    }

            # Находим ноты, звучащие одновременно (в той же позиции)
            if hasattr(current_note, 'offset'):
                current_offset = current_note.offset
                for i, other_note in enumerate(measure_notes):
                    if i != position and hasattr(other_note, 'offset'):
                        if abs(other_note.offset - current_offset) < 0.01:
                            overlap_info['simultaneous_notes'].append({
                                'note': str(other_note),
                                'position': i,
                                'interval': NoteAnalyzer._get_interval_between_notes(other_note, current_note)
                                if hasattr(other_note, 'pitch') and hasattr(current_note, 'pitch')
                                else None
                            })

        except Exception as e:
            overlap_info['error'] = str(e)

        return overlap_info

    @staticmethod
    def _notes_overlap(note1: note.Note, note2: note.Note) -> bool:
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

    @staticmethod
    def _determine_overlap_type(note1: note.Note, note2: note.Note) -> str:
        """Определяет тип перекрытия нот."""
        try:
            if not NoteAnalyzer._notes_overlap(note1, note2):
                return "no_overlap"

            end1 = note1.offset + note1.duration.quarterLength
            start2 = note2.offset

            overlap_amount = end1 - start2

            # Процент перекрытия относительно длительности первой ноты
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

    @staticmethod
    def _calculate_overlap_ratio(note1: note.Note, note2: note.Note) -> float:
        """Вычисляет коэффициент перекрытия."""
        try:
            if not NoteAnalyzer._notes_overlap(note1, note2):
                return 0.0

            end1 = note1.offset + note1.duration.quarterLength
            start2 = note2.offset

            overlap_amount = end1 - start2

            # Коэффициент относительно более короткой ноты
            duration1 = note1.duration.quarterLength
            duration2 = note2.duration.quarterLength

            return overlap_amount / min(duration1, duration2)

        except:
            return 0.0

    @staticmethod
    def _get_interval_between_notes(note1: note.Note, note2: note.Note) -> Dict[str, Any]:
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

                # Дополнительная информация
                interval_info['is_consonant'] = intv.isConsonant()
                interval_info['is_dissonant'] = intv.isDissonant()
                interval_info['is_perfect'] = intv.isPerfectType
                interval_info['generic'] = intv.generic.simpleUndirected

        except Exception as e:
            interval_info['error'] = str(e)

        return interval_info

    @staticmethod
    def _add_note_relations(notes_info: List[Dict[str, Any]]) -> None:
        """Добавляет информацию о связях между нотами."""
        # Группируем по партиям
        parts_dict = {}
        for note_info in notes_info:
            part_idx = note_info.get('part_index')
            if part_idx not in parts_dict:
                parts_dict[part_idx] = []
            parts_dict[part_idx].append(note_info)

        # Для каждой партии устанавливаем связи
        for part_notes in parts_dict.values():
            # Сортируем по такту и позиции
            part_notes.sort(key=lambda x: (x['measure'], x['position_in_measure']))

            for i in range(len(part_notes)):
                current_note = part_notes[i]

                # Связь с предыдущей нотой
                if i > 0:
                    prev_note = part_notes[i - 1]
                    current_note['relations'] = current_note.get('relations', {})
                    current_note['relations']['previous'] = {
                        'token': prev_note['token'],
                        'interval': NoteAnalyzer._get_interval_between_notes(
                            prev_note['note_object'],
                            current_note['note_object']
                        ) if (hasattr(prev_note['note_object'], 'pitch') and
                              hasattr(current_note['note_object'], 'pitch')) else None,
                        'time_gap': NoteAnalyzer._calculate_time_gap(
                            prev_note['note_object'],
                            current_note['note_object']
                        )
                    }

                # Связь со следующей нотой
                if i < len(part_notes) - 1:
                    next_note = part_notes[i + 1]
                    current_note['relations'] = current_note.get('relations', {})
                    current_note['relations']['next'] = {
                        'token': next_note['token'],
                        'interval': NoteAnalyzer._get_interval_between_notes(
                            current_note['note_object'],
                            next_note['note_object']
                        ) if (hasattr(current_note['note_object'], 'pitch') and
                              hasattr(next_note['note_object'], 'pitch')) else None
                    }

    @staticmethod
    def _calculate_time_gap(note1: note.Note, note2: note.Note) -> float:
        """Вычисляет временной промежуток между нотами."""
        try:
            if hasattr(note1, 'offset') and hasattr(note2, 'offset'):
                end1 = note1.offset + note1.duration.quarterLength
                start2 = note2.offset

                return max(0, start2 - end1)
        except:
            pass
        return 0.0

    @staticmethod
    def _get_pitch_spelling(pitch_obj: pitch.Pitch) -> Dict[str, Any]:
        """Возвращает информацию о написании высоты звука."""
        spelling = {
            'enharmonic_spellings': [],
            'preferred_spelling': None,
            'spelling_ambiguity': False
        }

        try:
            # Получаем альтернативные написания
            enharmonics = pitch_obj.getAllCommonEnharmonics()
            spelling['enharmonic_spellings'] = [str(p) for p in enharmonics]

            # Предпочтительное написание
            spelling['preferred_spelling'] = pitch_obj.nameWithOctave

            # Проверяем на неоднозначность
            spelling['spelling_ambiguity'] = len(enharmonics) > 1

        except:
            pass

        return spelling

    @staticmethod
    def _get_duration_components(duration_obj) -> List[Dict[str, Any]]:
        """Разбивает длительность на компоненты."""
        components = []

        try:
            # Для тюплетов
            if hasattr(duration_obj, 'tuplets') and duration_obj.tuplets:
                for tuplet in duration_obj.tuplets:
                    components.append({
                        'type': 'tuplet',
                        'ratio': str(tuplet.tupletRatio()),
                        'number_notes': tuplet.numberNotesActual,
                        'duration_notes': tuplet.numberNotesNormal
                    })

            # Основная длительность
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

    # Дополнительные утилитарные методы

    @staticmethod
    def get_note_statistics(notes_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Возвращает статистику по нотам."""
        stats = {
            'total_notes': len(notes_info),
            'by_type': {},
            'pitch_range': {'min': 127, 'max': 0},
            'average_duration': 0,
            'overlap_statistics': {},
            'part_distribution': {}
        }

        try:
            durations = []
            pitch_numbers = []

            for note_info in notes_info:
                # Распределение по типам
                note_type = note_info['characteristics']['basic']['type']
                stats['by_type'][note_type] = stats['by_type'].get(note_type, 0) + 1

                # Распределение по партиям
                part_idx = note_info.get('part_index')
                stats['part_distribution'][part_idx] = stats['part_distribution'].get(part_idx, 0) + 1

                # Статистика по высотам
                pitch_info = note_info['characteristics'].get('pitch_info', {})
                midi_num = pitch_info.get('midi_number')
                if midi_num is not None:
                    pitch_numbers.append(midi_num)

                # Статистика по длительностям
                rhythmic_info = note_info['characteristics'].get('rhythmic_info', {})
                duration = rhythmic_info.get('quarter_length')
                if duration is not None:
                    durations.append(duration)

                # Статистика по перекрытиям
                overlap_type = note_info.get('overlap_info', {}).get('previous_overlap', {}).get('overlap_type')
                if overlap_type:
                    stats['overlap_statistics'][overlap_type] = stats['overlap_statistics'].get(overlap_type, 0) + 1

            # Вычисляем общую статистику
            if pitch_numbers:
                stats['pitch_range']['min'] = min(pitch_numbers)
                stats['pitch_range']['max'] = max(pitch_numbers)
                stats['average_pitch'] = statistics.mean(pitch_numbers)

            if durations:
                stats['average_duration'] = statistics.mean(durations)
                stats['min_duration'] = min(durations)
                stats['max_duration'] = max(durations)

        except Exception as e:
            stats['error'] = str(e)

        return stats

    @staticmethod
    def export_to_json(notes_info: List[Dict[str, Any]], output_path: str) -> None:
        """Экспортирует информацию о нотах в JSON файл."""
        import json
        from datetime import datetime

        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_notes': len(notes_info),
                'analyzer_version': '1.0'
            },
            'notes': [],
            'statistics': NoteAnalyzer.get_note_statistics(notes_info),
            'intervals': NoteAnalyzer.get_note_intervals(notes_info)
        }

        # Упрощаем информацию о нотах для экспорта
        for note_info in notes_info:
            simplified = {
                'token': note_info['token'],
                'part_index': note_info['part_index'],
                'part_name': note_info['part_name'],
                'measure': note_info['measure'],
                'position_in_measure': note_info['position_in_measure'],
                'pitch': note_info['characteristics'].get('pitch_info', {}).get('name_with_octave'),
                'midi_number': note_info['characteristics'].get('pitch_info', {}).get('midi_number'),
                'duration': note_info['characteristics'].get('rhythmic_info', {}).get('quarter_length'),
                'is_rest': note_info['characteristics']['basic']['is_rest'],
                'is_chord': note_info['characteristics']['basic']['is_chord'],
                'overlap_type': note_info.get('overlap_info', {}).get('previous_overlap', {}).get('overlap_type'),
                'relations': note_info.get('relations', {})
            }
            export_data['notes'].append(simplified)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)