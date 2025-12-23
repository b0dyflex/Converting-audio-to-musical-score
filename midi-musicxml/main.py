import os
import sys
import argparse
from converter import MidiToMusicXMLConverter


def interactive_mode():
    """
    Интерактивный режим работы программы
    """
    print("\n" + "=" * 60)
    choice = input(
        "Выберите режим работы:\n1. Конвертация одного файла\n2. Пакетная конвертация\n\nВаш выбор (1/2): ").strip()

    if choice == '1':
        input_file = input("Введите путь к MIDI файлу: ").strip('"\' ')
        if input_file:
            return {'input_file': input_file}
        else:
            print("Не указан файл")
            return None
    elif choice == '2':
        directory = input("Введите путь к директории: ").strip('"\' ')
        if directory:
            args = {'directory': directory}
            recursive_choice = input("Рекурсивно обрабатывать поддиректории? (y/n): ").strip().lower()
            args['recursive'] = recursive_choice == 'y'
            return args
        else:
            print("Не указана директория")
            return None
    else:
        print("Неверный выбор")
        return None


def show_file_preview(file_path):
    """
    Показывает превью сконвертированного файла
    """
    try:
        if os.path.getsize(file_path) < 5000:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(300)
                print("Превью файла (первые 300 символов):")
                print("-" * 40)
                print(content)
                print("-" * 40)
    except Exception:
        pass


def open_result_directory(file_path):
    """
    Открывает папку с результатом конвертации
    """
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir and os.path.exists(output_dir):
            print(f"\nФайл сохранен в: {output_dir}")
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_dir}"')
            else:
                os.system(f'xdg-open "{output_dir}"')
    except Exception:
        pass


def parse_arguments():
    """
    Парсит аргументы командной строки
    """
    parser = argparse.ArgumentParser(
        description='Конвертер MIDI в MusicXML',
        formatter_class=argparse.RawDescriptionHelpFormatter)

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

    return parser.parse_args()


def main():
    """
    Основная функция с поддержкой командной строки
    """
    args = parse_arguments()

    converter = MidiToMusicXMLConverter()

    if args.all_files:
        converter.midi_extensions = []

    if not args.input_file and not args.directory:
        parser = argparse.ArgumentParser()
        parse_arguments()  # Чтобы показать help

        # Интерактивный режим
        interactive_args = interactive_mode()
        if interactive_args:
            if 'input_file' in interactive_args:
                args.input_file = interactive_args['input_file']
            else:
                args.directory = interactive_args['directory']
                args.recursive = interactive_args.get('recursive', False)
        else:
            return

    # Выполняем конвертацию
    if args.input_file:
        print("=" * 60)
        print("КОНВЕРТАЦИЯ MIDI В MUSICXML")
        print("=" * 60)

        result = converter.convert_single_file(args.input_file, args.output)

        if result:
            print("\n" + "=" * 60)
            print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
            print("=" * 60)

            show_file_preview(result)

            open_result_directory(result)

    elif args.directory:
        print("=" * 60)
        print("ПАКЕТНАЯ КОНВЕРТАЦИЯ MIDI В MUSICXML")
        print("=" * 60)

        converter.convert_batch(args.directory, args.output_dir, args.recursive)


if __name__ == "__main__":
    main()