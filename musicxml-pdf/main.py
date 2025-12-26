import subprocess
import os
import sys
from pathlib import Path
import time
import shutil

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
                    f"{pf}\\MuseScore 2\\bin\\MuseScore.exe",
                ])
        possible_paths.extend(["MuseScore4.exe", "MuseScore3.exe", "mscore", "musescore"])

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
            "/usr/bin/musescore4",
            "/usr/bin/musescore3",
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
    if path in ['musescore', 'mscore', 'musescore4', 'musescore3', 'MuseScore4.exe', 'MuseScore3.exe']:
        try:
            if sys.platform == 'win32':
                result = subprocess.run(['where', path],
                                        capture_output=True,
                                        text=True,
                                        timeout=5,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                result = subprocess.run(['which', path],
                                        capture_output=True,
                                        text=True,
                                        timeout=5)
            return result.returncode == 0
        except:
            return False
    else:
        return os.path.exists(path)


def normalize_path(path):
    """Нормализация пути для Windows"""
    path = str(path)
    if sys.platform == 'win32':
        # Заменяем слеши и убираем кавычки
        path = path.replace('/', '\\')
        path = path.strip('"').strip("'")
    return os.path.abspath(path)


def check_file_access(file_path):
    """Проверка доступа к файлу"""
    file_path = normalize_path(file_path)

    checks = []

    # Проверка существования
    if not os.path.exists(file_path):
        return False, f"Файл не существует: {file_path}"

    # Проверка доступа на чтение
    if not os.access(file_path, os.R_OK):
        return False, f"Нет доступа на чтение: {file_path}"

    # Проверка, что это файл, а не папка
    if not os.path.isfile(file_path):
        return False, f"Это папка, а не файл: {file_path}"

    # Проверка размера
    try:
        size = os.path.getsize(file_path)
        if size == 0:
            return False, f"Файл пустой: {file_path}"
        if size > 500 * 1024 * 1024:  # 500 MB
            return False, f"Файл слишком большой ({size / 1024 / 1024:.1f} MB)"
    except:
        return False, f"Не удалось проверить размер файла: {file_path}"

    return True, "OK"


def check_output_directory(output_path):
    """Проверка возможности записи в выходную директорию"""
    output_path = normalize_path(output_path)
    output_dir = os.path.dirname(output_path) or os.getcwd()

    # Проверяем существование директории
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except:
            return False, f"Не удалось создать директорию: {output_dir}"

    # Проверяем доступ на запись
    if not os.access(output_dir, os.W_OK):
        return False, f"Нет доступа на запись в директорию: {output_dir}"

    return True, "OK"


def convert_musicxml_to_pdf(input_path, output_path, musescore_path):
    """
    Конвертирует MusicXML файл в PDF используя MuseScore
    """
    # Нормализуем пути
    input_path = normalize_path(input_path)
    output_path = normalize_path(output_path)
    musescore_path = normalize_path(musescore_path)

    print(f"\nКонвертация: {Path(input_path).name}")
    print(f"Входной файл: {input_path}")
    print(f"Выходной файл: {output_path}")

    # Проверяем доступ к файлам
    input_check, input_msg = check_file_access(input_path)
    if not input_check:
        print(f"\n✗ Проблема с входным файлом: {input_msg}")
        return False

    output_check, output_msg = check_output_directory(output_path)
    if not output_check:
        print(f"\n✗ Проблема с выходной директорией: {output_msg}")
        return False

    # Проверяем MuseScore
    if not os.path.exists(musescore_path):
        print(f"\n✗ MuseScore не найден: {musescore_path}")
        return False

    try:
        print("\nЗапуск MuseScore...")
        start_time = time.time()

        # Базовая команда
        cmd = [musescore_path, "-o", output_path, input_path]

        # Показываем команду для отладки
        if len(' '.join(cmd)) < 200:  # Не показываем слишком длинные команды
            print(f"Команда: {' '.join(cmd)}")

        # Настройки для subprocess в зависимости от платформы
        creation_flags = 0
        if sys.platform == 'win32':
            creation_flags = subprocess.CREATE_NO_WINDOW

        # Запускаем конвертацию
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 час таймаут
            encoding='utf-8',
            errors='ignore',
            creationflags=creation_flags
        )

        elapsed = time.time() - start_time

        # Проверяем результат
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                pdf_size = os.path.getsize(output_path) / 1024
                print(f"\nУспешно создан: {output_path}")
                print(f"  Размер PDF: {pdf_size:.1f} KB")
                print(f"  Время конвертации: {elapsed // 60:.0f} мин {elapsed % 60:.1f} сек")
                return True
            else:
                print("\nPDF файл не создан или пустой")
                return False
        else:
            print(f"\nОшибка MuseScore (код: {result.returncode})")

            # Анализируем ошибку
            error_info = analyze_musescore_error(result.returncode, result.stderr, result.stdout)

            print("\nДетали ошибки:")
            print("-" * 50)
            for line in error_info.split('\n'):
                if line.strip():
                    print(f"  {line}")
            print("-" * 50)

            return False

    except subprocess.TimeoutExpired:
        print(f"\nТаймаут конвертации (1 час)")
        print("Возможно, файл слишком большой или MuseScore завис")
        return False

    except Exception as e:
        print(f"\nНеожиданная ошибка: {str(e)}")
        return False


def analyze_musescore_error(returncode, stderr, stdout):
    """Анализ ошибок MuseScore"""
    error_text = stderr or stdout or ""

    # Коды ошибок Windows
    if returncode == 1320:
        return "Ошибка 1320: Проблема с доступом к файлу или директории.\n" \
               "Возможные причины:\n" \
               "1. Файл открыт в другой программе\n" \
               "2. Нет прав на запись в папку\n" \
               "3. Слишком длинный путь к файлу\n" \
               "4. Специальные символы в пути"

    elif returncode == 5:
        return "Ошибка 5: Доступ запрещен.\n" \
               "Проверьте права доступа к файлам."

    elif returncode == 2:
        return "Ошибка 2: Файл не найден.\n" \
               "Проверьте путь к входному файлу."

    # Анализ текста ошибки
    error_lower = error_text.lower()

    if "permission" in error_lower or "access" in error_lower:
        return "Ошибка доступа. Проверьте:\n" \
               "1. Закройте файл в других программах\n" \
               "2. Запустите программу от имени администратора\n" \
               "3. Проверьте антивирус"

    elif "corrupt" in error_lower:
        return "Файл поврежден или имеет неверный формат."

    elif "memory" in error_lower:
        return "Нехватка памяти. Закройте другие программы."

    elif "timeout" in error_lower:
        return "Таймаут операции. Файл слишком большой."

    # Общий случай
    if error_text.strip():
        # Берем последние 3 строки ошибки
        lines = [line.strip() for line in error_text.split('\n') if line.strip()]
        last_lines = lines[-3:] if len(lines) > 3 else lines
        return '\n'.join(last_lines)

    return f"Код ошибки: {returncode}"


def get_input_file():
    """Упрощенная функция получения входного файла"""
    print("\n" + "=" * 60)
    print("   ВЫБОР ФАЙЛА")
    print("=" * 60)

    while True:
        print("\nВведите путь к MusicXML файлу или перетащите файл в окно:")
        print(f"Поддерживаемые форматы: {', '.join(SUPPORTED_EXTENSIONS)}")

        path = input("\nПуть к файлу: ").strip()

        if not path:
            print("Путь не может быть пустым.")
            continue

        # Очистка пути
        path = path.strip('"').strip("'")
        path = normalize_path(path)

        # Проверка существования
        if not os.path.exists(path):
            print(f"✗ Файл не найден: {path}")
            retry = input("Попробовать другой путь? (y/n): ").lower()
            if retry not in ['y', 'yes', 'д', 'да']:
                return None
            continue

        # Проверка расширения
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"⚠ Файл имеет нестандартное расширение: {ext}")
            proceed = input("Все равно продолжить? (y/n): ").lower()
            if proceed not in ['y', 'yes', 'д', 'да']:
                continue

        print(f"✓ Выбран файл: {path}")
        return path


def get_output_filename(input_file):
    """Получение имени выходного файла"""
    input_path = Path(input_file)

    # Создаем имя файла без пробелов и специальных символов
    safe_name = input_path.stem.replace(' ', '_').replace('(', '').replace(')', '')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '._-')
    default_output = input_path.with_name(safe_name).with_suffix('.pdf')

    while True:
        print(f"\nИмя выходного PDF файла:")
        print(f"По умолчанию: {default_output}")

        output_file = input("Введите имя файла или нажмите Enter для использования значения по умолчанию: ").strip()

        if not output_file:
            output_file = str(default_output)
        elif not output_file.lower().endswith('.pdf'):
            output_file += '.pdf'

        # Нормализуем путь
        output_file = normalize_path(output_file)

        # Проверяем директорию
        output_dir = os.path.dirname(output_file) or os.getcwd()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Создана директория: {output_dir}")
            except Exception as e:
                print(f"Не удалось создать директорию: {e}")
                continue

        # Проверка перезаписи
        if os.path.exists(output_file):
            overwrite = input(f"Файл '{Path(output_file).name}' уже существует. Перезаписать? (y/n): ").lower()
            if overwrite not in ['y', 'yes', 'д', 'да']:
                continue

        return output_file



def main():
    """Основная функция программы"""

    # Поиск MuseScore
    print("\nПоиск MuseScore...")
    musescore_path = find_musescore()

    if not musescore_path:
        print("\n✗ MuseScore не найден!")
        print("Установите MuseScore с https://musescore.org")

        # Ручной ввод пути
        print("\nМожете указать путь к MuseScore вручную:")
        print("Пример: C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe")

        while True:
            manual_path = input("\nПуть к MuseScore (или Enter для выхода): ").strip()

            if not manual_path:
                print("Программа завершена.")
                return

            manual_path = normalize_path(manual_path)

            if os.path.exists(manual_path):
                musescore_path = manual_path
                break
            else:
                print(f"Файл не найден: {manual_path}")

    print(f"\n✓ Найден MuseScore: {musescore_path}")

    # Проверяем MuseScore
    try:
        test_result = subprocess.run([musescore_path, "--version"],
                                     capture_output=True,
                                     text=True,
                                     timeout=10,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if test_result.returncode == 0:
            version_info = test_result.stdout.split('\n')[0][:100]
            print(f"Версия: {version_info}")
    except:
        print("⚠ Не удалось проверить версию MuseScore")

    success_count = 0
    total_count = 0

    while True:
        try:
            print("\n" + "=" * 60)

            # Получение входного файла
            input_file = get_input_file()
            if not input_file:
                print("Файл не выбран.")
                continue

            # Получение выходного файла
            output_file = get_output_filename(input_file)

            print("\n" + "=" * 60)
            print("НАЧАЛО КОНВЕРТАЦИИ")
            print("=" * 60)

            # Конвертация
            success = convert_musicxml_to_pdf(input_file, output_file, musescore_path)

            if success:
                success_count += 1
                print(f"\n✓ Конвертация успешна!")
            else:
                print(f"\n✗ Конвертация не удалась")

            total_count += 1

            # Продолжить?
            print("\n" + "-" * 60)
            again = input("Конвертировать еще один файл? (y/n): ").lower()
            if again not in ['y', 'yes', 'д', 'да']:
                break

        except KeyboardInterrupt:
            print("\n\nПрограмма завершена пользователем.")
            break

        except Exception as e:
            print(f"\n⚠ Ошибка: {e}")
            continue_choice = input("Продолжить? (y/n): ").lower()
            if continue_choice not in ['y', 'yes', 'д', 'да']:
                break

    # Статистика
    print("\n" + "=" * 60)
    print("СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего попыток: {total_count}")
    print(f"Успешно: {success_count}")
    if total_count > 0:
        print(f"Процент успеха: {success_count / total_count * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    try:
        # Устанавливаем кодировку для Windows
        if sys.platform == 'win32':
            import io

            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

        main()

    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()

    input("\nНажмите Enter для выхода...")