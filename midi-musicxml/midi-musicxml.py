import os
import sys
import argparse
from music21 import converter, environment, metadata


class MidiToMusicXMLConverter:
    def __init__(self):
        self.setup_environment()
        # Расширения MIDI файлов, которые поддерживаются
        self.midi_extensions = ['.mid', '.midi', '.mid', '.MID', '.MIDI']
        # Расширения файлов, которые music21 может прочитать как MIDI
        self.supported_extensions = self.midi_extensions + ['.kar', '.rmi']

    def setup_environment(self):
        """Настраивает окружение для music21"""
        try:
            us = environment.UserSettings()

            programs_to_check = [
                'musescore',
                'musescore4',
                'musescore3',
                'finale',
                'sibelius',
                'lilypond',
            ]

            for program in programs_to_check:
                if us[program]:
                    print(f"Найдена программа для нотной записи: {program}")
                    break
            else:
                print("  Не найдена программа для нотной записи. Установите MuseScore для лучших результатов.")
                print("   Скачайте с: https://musescore.org/")

        except Exception as e:
            print(f" Ошибка при настройке окружения: {e}")

    def is_midi_file(self, filepath):
        """Проверяет, является ли файл MIDI файлом по расширению"""
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.midi_extensions

    def convert_single_file(self, input_path, output_path=None):
        """
        Конвертирует один MIDI файл в MusicXML
        """
        try:
            if not os.path.exists(input_path):
                print(f"Файл не найден: {input_path}")
                return None

            # Проверяем расширение файла
            if not self.is_midi_file(input_path):
                print(f"Внимание: Файл не имеет стандартного расширения MIDI: {input_path}")
                print(f"Поддерживаемые расширения: {', '.join(self.midi_extensions)}")
                user_input = input("Продолжить конвертацию? (y/n): ").strip().lower()
                if user_input != 'y':
                    print("Конвертация отменена пользователем")
                    return None

            if output_path is None:
                base_name = os.path.splitext(input_path)[0]
                output_path = f"{base_name}.musicxml"
            elif not output_path.lower().endswith(('.musicxml', '.xml', '.mxl')):
                # Добавляем расширение если его нет
                output_path = f"{output_path}.musicxml"

            print(f" Входной файл: {input_path}")
            print(f" Выходной файл: {output_path}")

            # Загружаем MIDI файл
            print(" Загружаю MIDI файл...")
            try:
                score = converter.parse(input_path, format='midi')
            except Exception as parse_error:
                print(f"  Не удалось загрузить как MIDI, пробую общий парсер...")
                score = converter.parse(input_path)

            # Улучшаем партитуру
            print(" Обрабатываю партитуру...")
            score = self.enhance_score(score)

            # Экспортируем в MusicXML
            print(" Сохраняю MusicXML файл...")
            score.write('musicxml', fp=output_path)

            # Проверяем создание файла
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f" Файл успешно создан: {output_path}")
                print(f" Размер файла: {file_size} байт")
                return output_path
            else:
                print(" Ошибка: файл не создан")
                return None

        except Exception as e:
            print(f" Ошибка при конвертации: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_batch(self, input_dir, output_dir=None, recursive=False):
        """
        Пакетная конвертация всех MIDI файлов в директории
        """
        try:
            if not os.path.exists(input_dir):
                print(f" Директория не найдена: {input_dir}")
                return

            # Создаем выходную директорию если нужно
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Собираем список файлов
            midi_files = []
            if recursive:
                for root, dirs, files in os.walk(input_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        if self.is_midi_file(filepath):
                            midi_files.append(filepath)
            else:
                for file in os.listdir(input_dir):
                    filepath = os.path.join(input_dir, file)
                    if os.path.isfile(filepath) and self.is_midi_file(filepath):
                        midi_files.append(filepath)

            if not midi_files:
                print(" MIDI файлы не найдены")
                print(f" Ожидаемые расширения: {', '.join(self.midi_extensions)}")

                # Предлагаем пользователю попробовать найти другие файлы
                user_input = input("Хотите попробовать найти файлы с другими расширениями? (y/n): ").strip().lower()
                if user_input == 'y':
                    # Ищем любые файлы, которые music21 может попытаться прочитать
                    all_files = []
                    if recursive:
                        for root, dirs, files in os.walk(input_dir):
                            for file in files:
                                all_files.append(os.path.join(root, file))
                    else:
                        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                     if os.path.isfile(os.path.join(input_dir, f))]

                    print(f"\nНайдено файлов в директории: {len(all_files)}")

                    for filepath in all_files:
                        print(f"  {os.path.basename(filepath)}")

                    user_input = input("\nХотите попробовать конвертировать все файлы? (y/n): ").strip().lower()
                    if user_input == 'y':
                        midi_files = all_files

                if not midi_files:
                    return

            print(f" Найдено файлов для обработки: {len(midi_files)}")

            # Конвертируем файлы
            successful = 0
            failed = 0

            for i, midi_file in enumerate(midi_files, 1):
                print(f"\n{'=' * 60}")
                print(f" Обрабатываю файл {i}/{len(midi_files)}")
                print(f" Имя файла: {os.path.basename(midi_file)}")
                print(f"{'=' * 60}")

                # Определяем путь для выходного файла
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(midi_file))[0]
                    output_file = os.path.join(output_dir, f"{base_name}.musicxml")
                else:
                    output_file = None

                # Конвертируем
                result = self.convert_single_file(midi_file, output_file)

                if result:
                    successful += 1
                else:
                    failed += 1

            # Выводим статистику
            print("\n" + "=" * 60)
            print(" СТАТИСТИКА КОНВЕРТАЦИИ:")
            print("=" * 60)
            print(f" Успешно: {successful}")
            print(f" Ошибок: {failed}")
            print(f" Всего файлов: {len(midi_files)}")

        except Exception as e:
            print(f" Ошибка при пакетной конвертации: {e}")
            import traceback
            traceback.print_exc()

    def enhance_score(self, score):
        """
        Улучшает партитуру для лучшего отображения в MusicXML
        """
        try:
            # Добавляем метаданные если их нет
            if score.metadata is None or not score.metadata.title:
                if score.metadata is None:
                    score.insert(0, metadata.Metadata())
                score.metadata.title = 'Converted from MIDI'

            # Обрабатываем длительности нот
            for part in score.parts:
                for measure in part.getElementsByClass('Measure'):
                    # Упрощаем длительности нот
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

            # Добавляем тональность если её нет
            if not score.flat.getElementsByClass('Key'):
                from music21 import key
                try:
                    detected_key = score.analyze('key')
                    for part in score.parts:
                        first_measure = part.getElementsByClass('Measure')[0]
                        first_measure.insert(0, detected_key)
                except:
                    # По умолчанию До мажор
                    default_key = key.Key('C')
                    for part in score.parts:
                        first_measure = part.getElementsByClass('Measure')[0]
                        first_measure.insert(0, default_key)

            # Добавляем размер такта если его нет
            if not score.flat.getElementsByClass('TimeSignature'):
                from music21 import meter
                default_time = meter.TimeSignature('4/4')
                for part in score.parts:
                    first_measure = part.getElementsByClass('Measure')[0]
                    first_measure.insert(0, default_time)

            return score

        except Exception as e:
            print(f" Ошибка при улучшении партитуры: {e}")
            return score


def main():
    """Основная функция с поддержкой командной строки"""
    parser = argparse.ArgumentParser(
        description='Конвертер MIDI в MusicXML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python midi_to_musicxml.py input.mid
  python midi_to_musicxml.py input.mid -o output.musicxml
  python midi_to_musicxml.py -d /путь/к/папке -r
  python midi_to_musicxml.py -d /путь/к/папке -o /путь/к/выходной_папке -r

Поддерживаемые форматы MIDI:
  .mid, .midi, .MID, .MIDI (также пробует конвертировать другие форматы)
        '''
    )

    # Группа для единичной конвертации
    single_group = parser.add_argument_group('Единичная конвертация')
    single_group.add_argument(
        'input_file',
        nargs='?',
        help='Путь к MIDI файлу для конвертации'
    )
    single_group.add_argument(
        '-o', '--output',
        help='Путь для сохранения MusicXML файла'
    )

    # Группа для пакетной конвертации
    batch_group = parser.add_argument_group('Пакетная конвертация')
    batch_group.add_argument(
        '-d', '--directory',
        help='Директория с MIDI файлами для пакетной конвертации'
    )
    batch_group.add_argument(
        '-od', '--output-dir',
        help='Выходная директория для пакетной конвертации'
    )
    batch_group.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Рекурсивно обрабатывать поддиректории'
    )

    # Новый параметр для всех файлов
    parser.add_argument(
        '-a', '--all-files',
        action='store_true',
        help='Пробовать конвертировать все файлы, не только с MIDI расширениями'
    )

    # Дополнительные параметры
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод'
    )

    args = parser.parse_args()

    # Создаем конвертер
    converter = MidiToMusicXMLConverter()

    # Если включен режим "все файлы", расширяем список поддерживаемых расширений
    if args.all_files:
        converter.midi_extensions = []  # Будем пробовать все файлы

    # Проверяем аргументы
    if not args.input_file and not args.directory:
        parser.print_help()
        print("\n" + "=" * 60)

        # Интерактивный режим
        choice = input(
            "\nВыберите режим работы:\n1. Конвертация одного файла\n2. Пакетная конвертация\n\nВаш выбор (1/2): ").strip()

        if choice == '1':
            input_file = input("Введите путь к MIDI файлу: ").strip('"\' ')
            if input_file:
                args.input_file = input_file
            else:
                print("Не указан файл")
                return
        elif choice == '2':
            directory = input("Введите путь к директории: ").strip('"\' ')
            if directory:
                args.directory = directory
            else:
                print("Не указана директория")
                return
            recursive_choice = input("Рекурсивно обрабатывать поддиректории? (y/n): ").strip().lower()
            args.recursive = recursive_choice == 'y'
        else:
            print("Неверный выбор")
            return

    # Выполняем конвертацию
    if args.input_file:
        # Единичная конвертация
        print("=" * 60)
        print("КОНВЕРТАЦИЯ MIDI В MUSICXML")
        print("=" * 60)

        result = converter.convert_single_file(args.input_file, args.output)

        if result:
            print("\n" + "=" * 60)
            print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
            print("=" * 60)

            # Показываем превью файла
            try:
                if os.path.getsize(result) < 5000:
                    with open(result, 'r', encoding='utf-8') as f:
                        content = f.read(300)
                        print("Превью файла (первые 300 символов):")
                        print("-" * 40)
                        print(content)
                        print("-" * 40)
            except:
                pass

            # Открываем папку с результатом
            try:
                output_dir = os.path.dirname(result)
                if output_dir and os.path.exists(output_dir):
                    print(f"\nФайл сохранен в: {output_dir}")
                    if sys.platform == 'win32':
                        os.startfile(output_dir)
                    elif sys.platform == 'darwin':
                        os.system(f'open "{output_dir}"')
                    else:
                        os.system(f'xdg-open "{output_dir}"')
            except:
                pass

    elif args.directory:
        # Пакетная конвертация
        print("=" * 60)
        print("ПАКЕТНАЯ КОНВЕРТАЦИЯ MIDI В MUSICXML")
        print("=" * 60)

        converter.convert_batch(args.directory, args.output_dir, args.recursive)


if __name__ == "__main__":
    main()