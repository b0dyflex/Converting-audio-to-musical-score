import os
from music21 import converter, metadata, key, meter
from music21.stream import Score, Part, Opus
from pathlib import Path


class MidiToMusicXMLConverter:
    def __init__(self) -> None:
        self.midi_extensions = ['.mid', '.midi', '.mid', '.MID', '.MIDI']

    def is_midi_file(self, filepath: str) -> bool:
        """Проверяет, является ли файл MIDI файлом по расширению."""
        return Path(filepath).suffix in self.midi_extensions

    def convert_file(self, input_path: str, output_path: str = None):
        """Конвертирует MIDI файл в MusicXML."""
        try:
            if not os.path.exists(input_path):
                raise ValueError("No such file")
            if not self.is_midi_file(input_path):
                raise ValueError(f"Unsupported file extension, support only {self.midi_extensions}")
            score = self.parse(input_path)
            enchanced_score = ScoringEnchancer.enhance_score(score)
            print(f"output_path: {self.export_music_xml(output_path, enchanced_score)}")
        except Exception as e:
            raise e

    def parse(self, input_path: str) -> Score | Part | Opus:
        """Чтение записи."""
        try:
            return converter.parse(input_path, format='midi')
        except Exception:
            print("Failed to load Midi, try common parser.")
            return converter.parse(input_path)

    def export_music_xml(self, output_path: str, enchanced_score: Score | Part | Opus) -> str:
        """Запись musicxml файла."""
        try:
            enchanced_score.write('musicxml', fp=output_path)
            if os.path.exists(output_path):
                return output_path
            raise ValueError('File not created')
        except Exception as e:
            print(e)

    def convert_batch(self, input_dir: str, output_dir: str = None) -> None:
        """Пакетная конвертация всех MIDI файлов в директории."""
        try:
            if not os.path.exists(input_dir):
                print(f'No such directory: {input_dir}')
                return None
            if output_dir and not os.path.exists(output_dir):
                print('No output_directory, creating...')
                os.makedirs(output_dir)
            midi_files = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    if self.is_midi_file(filepath):
                        midi_files.append(filepath)
            if not midi_files:
                print('No such midi_files')
            successful = 0
            failed = 0
            print(f"\n{'=' * 60}")
            for i, midi_file in enumerate(midi_files, 1):
                print(f" Обрабатываю файл {i}/{len(midi_files)}")
                print(f" Имя файла: {os.path.basename(midi_file)}")
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(midi_file))[0]
                    output_file = os.path.join(output_dir, f"{base_name}.musicxml")
                else:
                    output_file = None
                result = self.convert_file(midi_file, output_file)
                if result:
                    successful += 1
                else:
                    failed += 1
            print(f"{'=' * 60}")
            # Выводим статистику
            print("\n" + "=" * 60)
            print(" СТАТИСТИКА КОНВЕРТАЦИИ:")
            print("=" * 60)
            print(f" Успешно: {successful}")
            print(f" Ошибок: {failed}")
            print(f" Всего файлов: {len(midi_files)}")
        except Exception as e:
            print(f" Ошибка при пакетной конвертации: {e}")

class ScoringEnchancer:
    """Занимается улучшением партитуры для наилучшего отображения MusicXML."""

    @classmethod
    def _add_metadata(cls, score: Score | Part | Opus) -> None:
        """Добавить метаданные, если их нет."""
        if score.metadata is None or not score.metadata.title:
                if score.metadata is None:
                    score.insert(0, metadata.Metadata())
                score.metadata.title = 'Converted from MIDI'

    @classmethod
    def _prepare_note_duration(cls, score: Score | Part | Opus) -> None:
        """Обработать длительность нот."""
        for part in score.parts:
            for measure in part.getElementsByClass('Measure'):
                notes = measure.flatten().notes
                for note in notes:
                    quarter_length = note.quarterLength
                    if quarter_length < 0.5:
                        note.quarterLength = 0.25
                    elif quarter_length < 1:
                        note.quarterLength = 0.5
                    elif quarter_length < 2:
                        note.quarterLength = 1
                    elif quarter_length < 4:
                        note.quarterLength = 2
                    else:
                        note.quarterLength = 4

    @classmethod
    def _add_music_key(cls, score: Score | Part | Opus) -> None:
        """Добавить тональность, если её нет."""
        if not score.flatten().getElementsByClass('Key'):
            try:
                detected_key = score.analyze('key')
                for part in score.parts:
                    first_measure = part.getElementsByClass('Measure')[0]
                    first_measure.insert(0, detected_key)
            except Exception:
                default_key = key.Key('C') # До мажор
                for part in score.parts:
                    first_measure = part.getElementsByClass('Measure')[0]
                    first_measure.insert(0, default_key)
    
    @classmethod
    def _add_time_signature(cls, score: Score | Part | Opus) -> None:
        """Добавить размер такта, если его нет."""
        if not score.flatten().getElementsByClass('TimeSignature'):
            default_time = meter.TimeSignature('4/4')
            for part in score.parts:
                first_measure = part.getElementsByClass('Measure')[0]
                first_measure.insert(0, default_time)

    @classmethod
    def enhance_score(cls, score: Score | Part | Opus):
        """Улучшает партитуру для лучшего отображения в MusicXML."""
        try:
            if score.metadata is None or not score.metadata.title:
                cls._add_metadata(score)
            cls._prepare_note_duration(score)
            if not score.flatten().getElementsByClass('Key'):
                cls._add_music_key(score)
            if not score.flatten().getElementsByClass('TimeSignature'):
                cls._add_time_signature(score)
            return score

        except Exception as e:
            print(f" Ошибка при улучшении партитуры: {e}")
            return score