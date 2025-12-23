import subprocess
import os
import sys
from pathlib import Path
import glob

SUPPORTED_EXTENSIONS = ['.musicxml', '.xml', '.mxl']

def find_musescore():
    """Поиск MuseScore в системе"""
    possible_paths = []

    # Windows
    if sys.platform == 'win32':
        program_files = [
            os.environ.get('ProgramFiles', 'C:\\Program Files'),
            os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'),
        ]

        for pf in program_files:
            if pf:
                possible_paths.extend([
                    f"{pf}\\MuseScore 4\\bin\\MuseScore4.exe",
                    f"{pf}\\MuseScore 3\\bin\\MuseScore3.exe",
                    f"{pf}\\MuseScore 2\\bin\\MuseScore.exe"
                ])
        possible_paths.extend(["mscore", "musescore"])

    # macOS
    elif sys.platform == 'darwin':
        possible_paths.extend([
            "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
            "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
            "/Applications/MuseScore 2.app/Contents/MacOS/mscore",
            "/usr/local/bin/musescore",
            "/opt/homebrew/bin/musescore"
        ])

    # Linux
    else:
        possible_paths.extend([
            "/usr/bin/musescore",
            "/usr/bin/mscore",
            "/usr/local/bin/musescore",
            "musescore",
            "mscore"
        ])

    # Проверяем каждый путь
    for path in possible_paths:
        if check_program_exists(path):
            return path

    return None


def check_program_exists(path):
    """Проверка существования программы"""
    if path in ['musescore', 'mscore']:
        try:
            if sys.platform == 'win32':
                result = subprocess.run(['where', path],
                                        capture_output=True,
                                        text=True,
                                        timeout=2)
            else:
                result = subprocess.run(['which', path],
                                        capture_output=True,
                                        text=True,
                                        timeout=2)
            return result.returncode == 0
        except:
            return False
    else:
        return os.path.exists(path)


def convert_musicxml_to_pdf(input_path, output_path, musescore_path):
    """
    Конвертирует MusicXML файл в PDF используя MuseScore
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    print(f"\nКонвертация: {input_path} -> {output_path}")

    try:
        # Формируем команду
        cmd = [musescore_path, "-o", str(output_path), input_path]
        print(f"Использую: {musescore_path}")

        # Запускаем процесс конвертации
        print("Конвертация... (может занять некоторое время)")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            if os.path.exists(output_path):
                size = os.path.getsize(output_path) / 1024
                print(f"\nУспешно создан: {output_path}")
                print(f"Размер PDF: {size:.1f} KB")
                return True
            else:
                print("⚠ PDF файл не найден после конвертации")
                return False
        else:
            print(f"\nОшибка MuseScore (код: {result.returncode}):")
            if result.stderr:
                error_msg = result.stderr[:300]
                print(f"Ошибка: {error_msg}")
            return False

    except subprocess.TimeoutExpired:
        print("Таймаут конвертации")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False


# ==================== ФУНКЦИИ ДЛЯ ВВОДА ФАЙЛОВ ====================

def get_file_input_method():
    """Выбор способа ввода файла"""
    print("\n" + "=" * 60)
    print("   ВЫБОР ФАЙЛА")
    print("=" * 60)

    print("\nВыберите способ ввода файла:")
    print("1. Ввести путь вручную (или перетащить файл)")
    print("2. Выбрать из файлов в текущей папке")
    print("3. Поиск файлов по имени")

    while True:
        choice = input("\nВаш выбор (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Неверный выбор. Введите 1, 2 или 3.")


def manual_file_input():
    """Ручной ввод пути к файлу"""
    print("\nВведите путь к MusicXML файлу или перетащите файл в окно терминала.")
    print(f"Поддерживаемые форматы: {', '.join(SUPPORTED_EXTENSIONS)}")

    while True:
        path = input("\nПуть к файлу: ").strip()

        # Удаление кавычек для drag-and-drop
        path = path.strip('"').strip("'")

        if not path:
            print("Путь не может быть пустым.")
            continue

        if os.path.exists(path):
            # Проверяем расширение
            ext = Path(path).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                return path
            else:
                print(f"⚠ Файл имеет нестандартное расширение: {ext}")
                proceed = input("Все равно продолжить? (y/n): ").lower()
                if proceed in ['y', 'yes', 'д', 'да']:
                    return path
        else:
            print(f"Файл не найден: {path}")
            retry = input("Попробовать другой путь? (y/n): ").lower()
            if retry not in ['y', 'yes', 'д', 'да']:
                return None


def browse_current_directory():
    """Просмотр файлов в текущей папке"""
    current_dir = os.getcwd()
    print(f"\nТекущая папка: {current_dir}")

    # Ищем MusicXML файлы
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(f"*{ext}"))

    if not files:
        print("Не найдено MusicXML файлов в текущей папке.")
        return None

    # Сортируем файлы
    files.sort()

    print(f"\nНайдено {len(files)} файлов:")
    print("-" * 50)

    for i, file in enumerate(files, 1):
        size = os.path.getsize(file) / 1024  # Размер в KB
        filename = Path(file).name
        print(f"{i:3}. {filename:35} ({size:.1f} KB)")

    print("-" * 50)

    while True:
        try:
            choice = input("\nВыберите номер файла (или 0 для отмены): ").strip()

            if choice == '0':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
                print(f"Выбран файл: {selected_file}")
                return selected_file
            else:
                print(f"Неверный номер. Введите число от 1 до {len(files)}")

        except ValueError:
            print("Пожалуйста, введите число")


def search_files():
    """Поиск файлов по имени или маске"""
    print("\nПоиск файлов по имени или маске")
    print("Примеры: '*.musicxml', 'score*', '*part*.xml'")

    pattern = input("\nВведите маску для поиска: ").strip()

    if not pattern:
        pattern = "*.musicxml"
        print(f"Используется маска по умолчанию: {pattern}")

    files = glob.glob(pattern)

    if not files:
        print(f"Не найдено файлов по маске: {pattern}")
        return None

    # Фильтруем только поддерживаемые расширения
    filtered_files = [
        f for f in files
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS or
           input(f"Файл '{f}' имеет нестандартное расширение. Включить его? (y/n): ").lower() in ['y', 'yes', 'д', 'да']
    ]

    if not filtered_files:
        print("Нет файлов с поддерживаемыми расширениями.")
        return None

    print(f"\nНайдено {len(filtered_files)} файлов:")
    for i, file in enumerate(filtered_files[:20], 1):
        print(f"{i:3}. {file}")

    if len(filtered_files) > 20:
        print(f"... и еще {len(filtered_files) - 20} файлов")

    while True:
        try:
            choice = input("\nВыберите номер файла (или 0 для отмены): ").strip()

            if choice == '0':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(filtered_files):
                return filtered_files[idx]
            else:
                print(f"Неверный номер. Введите число от 1 до {len(filtered_files)}")
        except ValueError:
            print("Пожалуйста, введите число")


def get_input_file():
    """Основная функция получения входного файла"""
    method = get_file_input_method()

    if method == '1':
        return manual_file_input()
    elif method == '2':
        return browse_current_directory()
    elif method == '3':
        return search_files()

    return None


def get_output_filename(input_file):
    """Получение имени выходного файла"""
    input_path = Path(input_file)
    default_output = input_path.with_suffix('.pdf')

    while True:
        output_file = input(f"\nИмя PDF файла [Enter для '{default_output}']: ").strip()

        if not output_file:
            output_file = str(default_output)
            break

        # Добавляем расширение .pdf если его нет
        if not output_file.lower().endswith('.pdf'):
            output_file += '.pdf'
            print(f"Добавлено расширение .pdf: {output_file}")

        # Проверяем, не перезаписываем ли мы существующий файл
        if os.path.exists(output_file):
            overwrite = input(f"Файл '{output_file}' уже существует. Перезаписать? (y/n): ").lower()
            if overwrite in ['y', 'yes', 'д', 'да']:
                break
        else:
            break

    return output_file


# ==================== ФУНКЦИИ ИНТЕРФЕЙСА ====================

def show_header():
    """Показать заголовок программы"""
    print("\n" + "=" * 60)
    print("   КОНВЕРТЕР MusicXML В PDF ")
    print("=" * 60)
    print("\nКонвертирует нотные записи из MusicXML в формат PDF")
    print(f"Поддерживаемые форматы: {', '.join(SUPPORTED_EXTENSIONS)}")


def show_conversion_progress(input_file, output_file):
    """Показать информацию о процессе конвертации"""
    print("\n" + "=" * 60)
    print("   НАЧАЛО КОНВЕРТАЦИИ")
    print("=" * 60)
    print(f"Входной файл: {input_file}")
    print(f"Выходной файл: {output_file}")


def ask_to_open_pdf(output_file):
    """Спросить, открыть ли PDF файл после конвертации"""
    if os.path.exists(output_file):
        open_it = input("\nОткрыть PDF файл? (y/n): ").lower()
        if open_it in ['y', 'yes', 'д', 'да']:
            try:
                if sys.platform == 'win32':
                    os.startfile(output_file)
                elif sys.platform == 'darwin':
                    subprocess.run(['open', output_file])
                else:
                    subprocess.run(['xdg-open', output_file])
                print("PDF файл открыт.")
            except Exception as e:
                print(f"Не удалось открыть PDF: {e}")


def ask_to_continue():
    """Спросить, продолжать ли работу"""
    print("\n" + "-" * 60)
    again = input("Конвертировать еще один файл? (y/n): ").lower()
    return again in ['y', 'yes', 'д', 'да']


def show_statistics(success_count, total_count):
    """Показать статистику конвертаций"""
    print(f"\nСтатистика: {success_count}/{total_count} успешных конвертаций")


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    """Основная функция программы"""
    show_header()

    # Поиск MuseScore
    print("\nПоиск MuseScore...")
    musescore_path = find_musescore()

    if not musescore_path:
        print("\nMuseScore не найден!")
        print("Пожалуйста, установите MuseScore с https://musescore.org")
        print("\nПосле установки:")
        print("1. Запустите программу снова")
        print("2. Или укажите путь к MuseScore вручную")

        manual_path = input("\nВведите путь к MuseScore вручную (или Enter для выхода): ").strip()
        manual_path = manual_path.strip('"').strip("'")

        if manual_path and os.path.exists(manual_path):
            musescore_path = manual_path
        else:
            print("\nПрограмма завершена.")
            return

    print(f"Найден MuseScore: {musescore_path}")

    # Статистика
    success_count = 0
    total_count = 0

    # Главный цикл программы
    while True:
        try:
            total_count += 1

            # Получение входного файла
            input_file = get_input_file()

            if not input_file:
                print("Файл не выбран.")
                continue

            # Получение имени выходного файла
            output_file = get_output_filename(input_file)

            # Показ информации о конвертации
            show_conversion_progress(input_file, output_file)

            # Конвертация
            success = convert_musicxml_to_pdf(input_file, output_file, musescore_path)

            if success:
                success_count += 1
                print(f"\nКонвертация #{total_count} успешна!")

                # Предложение открыть PDF
                ask_to_open_pdf(output_file)
            else:
                print(f"\nКонвертация #{total_count} не удалась")

            # Показать статистику
            show_statistics(success_count, total_count)

            # Спросить о продолжении
            if not ask_to_continue():
                print(f"Всего конвертаций: {total_count}, успешно: {success_count}")
                break

        except KeyboardInterrupt:
            print("\n\nПрограмма завершена пользователем.")
            show_statistics(success_count, total_count)
            break

        except Exception as e:
            print(f"\nКритическая ошибка: {e}")

            continue_choice = input("\nПродолжить работу программы? (y/n): ").lower()
            if continue_choice not in ['y', 'yes', 'д', 'да']:
                show_statistics(success_count, total_count)
                break




if __name__ == "__main__":
    main()