import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from music21 import converter, environment, metadata, stream
import argparse
import traceback
import time


class MusicXMLtoPDFConverter:
    DEFAULT_LILYPOND_PATH = r"C:\Program Files\LilyPond\bin\lilypond.exe"

    def __init__(self, lilypond_path: Optional[str] = None) -> None:
        self.env: Optional[environment.Environment] = None
        self.lilypond_path: Optional[str] = None

        # Используем путь по умолчанию, если не указан другой
        if lilypond_path is None:
            lilypond_path = self.DEFAULT_LILYPOND_PATH

        self.setup_lilypond_environment(lilypond_path)

    def setup_lilypond_environment(self, lilypond_path: str) -> None:
        print(f"Настройка LilyPond по пути: {lilypond_path}")

        # Проверяем существование файла
        if not os.path.exists(lilypond_path):
            raise FileNotFoundError(
                f"LilyPond не найден по указанному пути: {lilypond_path}\n"
                f"Убедитесь, что LilyPond установлен по адресу: {lilypond_path}\n"
                "Или укажите правильный путь при создании конвертера."
            )

        # Проверяем, что это файл (не директория)
        if not os.path.isfile(lilypond_path):
            raise FileNotFoundError(
                f"Указанный путь ведёт к директории, а не к файлу: {lilypond_path}"
            )

        # Проверяем расширение .exe для Windows
        if sys.platform == "win32" and not lilypond_path.lower().endswith('.exe'):
            print(f"Предупреждение: Путь не заканчивается на .exe: {lilypond_path}")

        # Проверяем, что LilyPond можно запустить
        self._verify_lilypond_executable(lilypond_path)

        # Настраиваем окружение music21
        env = environment.Environment()
        env['lilypondPath'] = lilypond_path
        self.env = env
        self.lilypond_path = lilypond_path

        print(f"✓ LilyPond успешно настроен: {lilypond_path}")

    def _verify_lilypond_executable(self, path: str) -> None:
        try:
            # Нормализуем путь
            normalized_path = os.path.normpath(path)

            print(f"Проверка LilyPond: {normalized_path}")

            # Запускаем lilypond --version для проверки
            result = subprocess.run(
                [normalized_path, '--version'],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            # Извлекаем версию из вывода
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    version_line = lines[0]
                    print(f"✓ LilyPond версия: {version_line}")
                else:
                    print("✓ LilyPond запущен успешно (версия не определена)")

            # Проверяем наличие ошибок
            if result.stderr:
                print(f"Предупреждение от LilyPond: {result.stderr[:200]}...")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Таймаут при проверке LilyPond. Возможно, программа зависла.\n"
                "Проверьте установку LilyPond."
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else "Неизвестная ошибка"
            raise RuntimeError(
                f"Ошибка при запуске LilyPond (код {e.returncode}):\n"
                f"{error_msg[:500]}"
            )
        except FileNotFoundError:
            # Предлагаем альтернативные пути для поиска
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
                f"Нет прав на выполнение файла: {path}\n"
                f"Попробуйте запустить программу с правами администратора."
            )
        except Exception as e:
            raise RuntimeError(
                f"Неизвестная ошибка при проверке LilyPond:\n"
                f"{type(e).__name__}: {str(e)}"
            )

    def _get_alternative_lilypond_paths(self) -> List[str]:
        """Возвращает список альтернативных путей для поиска LilyPond."""
        alternative_paths = []

        if sys.platform == "win32":
            # Стандартные пути установки LilyPond на Windows
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

        # Добавляем поиск в PATH
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for dir_path in path_dirs:
            exe_path = os.path.join(dir_path, "lilypond.exe")
            if os.path.exists(exe_path):
                alternative_paths.append(exe_path)

        # Убираем дубликаты и несуществующие пути
        unique_paths = []
        for path in alternative_paths:
            if path not in unique_paths and os.path.exists(path):
                unique_paths.append(path)

        return unique_paths

    def convert_file(self, input_path: str, output_path: Optional[str] = None,
                     scale_factor: float = 0.8, add_page_numbers: bool = True,
                     **kwargs: Any) -> str:
        start_time = time.time()

        try:
            # Проверяем входной файл
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Входной файл не найден: {input_path}")

            if input_file.suffix.lower() not in ['.xml', '.mxl', '.musicxml']:
                print(f"Предупреждение: Нестандартное расширение файла: {input_file.suffix}")

            # Определяем выходной путь
            if output_path is None:
                output_file = input_file.with_suffix('.pdf')
            else:
                output_file = Path(output_path)
                # Если указана директория, создаём там файл с тем же именем
                if output_file.is_dir():
                    output_file = output_file / input_file.with_suffix('.pdf').name

            print(f"\nКонвертация: {input_file.name} -> {output_file.name}")
            print("-" * 50)

            # Шаг 1: Загрузка MusicXML файла
            print("1. Чтение MusicXML файла...")
            score = converter.parse(str(input_file))
            print(f"   ✓ Файл загружен: {input_file.name}")

            # Добавляем метаданные
            if score.metadata is None:
                score.metadata = metadata.Metadata()
            if not score.metadata.title:
                score.metadata.title = input_file.stem

            # Шаг 2: Создание LilyPond файла с настройками
            print("2. Подготовка LilyPond файла...")

            # Настройки LilyPond
            lilypond_settings = f"""
\\layout {{
    indent = #0
    short-indent = #0
    \\context {{
        \\Score
        \\override SpacingSpanner.common-shortest-duration = #(ly:make-moment 1/16)
    }}
}}

\\paper {{
    paper-width = {210 * scale_factor} \\mm
    paper-height = {297 * scale_factor} \\mm
    top-margin = 15 \\mm
    bottom-margin = 15 \\mm
    left-margin = 15 \\mm
    right-margin = 15 \\mm
    print-page-number = {"#t" if add_page_numbers else "#f"}
}}
"""

            # Создаём временный LilyPond файл
            temp_ly_file = output_file.with_suffix('.temp.ly')

            # Сначала создаём LilyPond файл
            score.write('lilypond', fp=str(temp_ly_file))

            # Читаем созданный файл и добавляем настройки
            with open(temp_ly_file, 'r', encoding='utf-8') as f:
                ly_content = f.read()

            # Вставляем настройки после версии LilyPond
            if '\\version' in ly_content:
                parts = ly_content.split('\\version', 1)
                version_line = '\\version' + parts[1].split('\n', 1)[0]
                rest_content = parts[1].split('\n', 1)[1] if len(parts[1].split('\n', 1)) > 1 else ''
                new_content = parts[0] + version_line + '\n' + lilypond_settings + rest_content
            else:
                new_content = lilypond_settings + '\n' + ly_content

            # Записываем обновлённый файл
            with open(temp_ly_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"   ✓ LilyPond файл подготовлен: {temp_ly_file.name}")

            # Шаг 3: Компиляция в PDF
            print("3. Компиляция PDF с помощью LilyPond...")

            # Подготавливаем команду для LilyPond
            output_dir = output_file.parent
            output_stem = output_file.stem

            cmd = [
                self.lilypond_path,
                '--pdf',
                '-o', str(output_dir / output_stem),
                str(temp_ly_file)
            ]

            # Запускаем компиляцию
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            # Отображаем прогресс
            print("   ", end='', flush=True)
            for i in range(10):
                time.sleep(0.2)
                print(".", end='', flush=True)

            # Ждём завершения
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = stderr if stderr else "Неизвестная ошибка"
                # Удаляем временный файл
                if temp_ly_file.exists():
                    temp_ly_file.unlink()
                raise RuntimeError(f"Ошибка компиляции LilyPond:\n{error_msg[:500]}")

            print("\n   ✓ PDF успешно скомпилирован")

            # Удаляем временный .ly файл
            if temp_ly_file.exists():
                temp_ly_file.unlink()
                print(f"   ✓ Временный файл удален: {temp_ly_file.name}")

            # Удаляем другие временные файлы
            for ext in ['.log', '.tex', '.count']:
                temp_file = output_dir / f"{output_stem}{ext}"
                if temp_file.exists():
                    temp_file.unlink()

            # Проверяем, что PDF создан
            if not output_file.exists():
                # Иногда LilyPond создаёт файлы с другим именем
                pdf_files = list(output_dir.glob(f"{output_stem}*.pdf"))
                if pdf_files:
                    actual_pdf = pdf_files[0]
                    if actual_pdf != output_file:
                        actual_pdf.rename(output_file)
                else:
                    raise FileNotFoundError(f"PDF файл не был создан")

            elapsed_time = time.time() - start_time
            file_size = output_file.stat().st_size / 1024  # в КБ

            print("-" * 50)
            print(f"✓ Конвертация завершена успешно!")
            print(f"  Время выполнения: {elapsed_time:.1f} сек")
            print(f"  Размер PDF: {file_size:.1f} КБ")
            print(f"  PDF файл: {output_file}")

            return str(output_file)

        except Exception as e:
            print(f"\n✗ Ошибка при конвертации:")
            print(f"  {type(e).__name__}: {str(e)}")
            raise

    def batch_convert(self, input_files: List[str], output_dir: Optional[str] = None) -> List[str]:
        """
        Пакетная конвертация нескольких файлов.

        Args:
            input_files: Список путей к MusicXML файлам
            output_dir: Директория для сохранения PDF файлов

        Returns:
            Список путей к созданным PDF файлам
        """
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

                result = self.convert_file(input_file, output_file)
                results.append(result)
                print(f"   ✓ Успешно")

            except Exception as e:
                print(f"   ✗ Ошибка: {str(e)[:100]}")
                results.append(None)

        # Статистика
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

        # Получаем версию LilyPond
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
    print("MUSICXML В PDF КОНВЕРТЕР")
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
            print("4. Выход")

            choice = input("\nВыберите действие (1-4): ").strip()

            if choice == '1':  # Конвертировать один файл
                try:
                    input_file = get_file_from_user()
                    output_file = get_output_path_suggestion(input_file)

                    # Дополнительные настройки
                    print("\nДополнительные настройки:")
                    scale = input(f"Масштаб (0.5-1.5, по умолчанию 0.8): ").strip()
                    scale_factor = float(scale) if scale else 0.8

                    page_nums = input("Добавить номера страниц? (y/n, по умолчанию y): ").strip().lower()
                    add_page_numbers = page_nums != 'n'

                    # Выполняем конвертацию
                    print("\nНачало конвертации...")
                    result = converter.convert_file(
                        input_file,
                        output_file,
                        scale_factor=scale_factor,
                        add_page_numbers=add_page_numbers
                    )

                    print(f"\n✓ Конвертация завершена!")
                    print(f"  Результат: {result}")

                    # Предлагаем открыть файл
                    open_file = input("\nОткрыть PDF файл? (y/n): ").strip().lower()
                    if open_file == 'y':
                        os.startfile(result) if sys.platform == "win32" else subprocess.run(['open', result])

                except Exception as e:
                    print(f"\n✗ Ошибка: {str(e)}")
                    print("Попробуйте снова.")

            elif choice == '2':  # Пакетная конвертация
                try:
                    files = get_batch_files_from_user()
                    if files:
                        output_dir = input("\nВведите директорию для сохранения (Enter для текущей): ").strip()
                        output_dir = output_dir if output_dir else None

                        print("\nНачало пакетной конвертации...")
                        results = converter.batch_convert(files, output_dir)

                        print("\nПакетная конвертация завершена!")

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

            elif choice == '4':  # Выход
                print("\nСпасибо за использование программы!")
                print("До свидания!")
                break

            else:
                print("Неверный выбор. Пожалуйста, выберите 1-4.")

    except FileNotFoundError as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("\nДля решения проблемы:")
        print("1. Установите LilyPond с https://lilypond.org/download.html")
        print("2. Убедитесь, что установка в папку: C:\\Program Files\\LilyPond")
        print("3. Или укажите путь вручную при запуске:")
        print("   python script.py --lilypond 'ваш\\путь\\lilypond.exe'")

    except Exception as e:
        print(f"\n✗ НЕИЗВЕСТНАЯ ОШИБКА: {type(e).__name__}: {e}")
        print("\nДетальная информация об ошибке:")
        traceback.print_exc()


def main():
    """Точка входа в программу."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Конвертер MusicXML в PDF')
    parser.add_argument('--lilypond', type=str, help='Путь к LilyPond')
    parser.add_argument('--input', type=str, help='Путь к входному файлу (без меню)')
    parser.add_argument('--output', type=str, help='Путь к выходному файлу')
    parser.add_argument('--batch', action='store_true', help='Пакетный режим')
    parser.add_argument('--scale', type=float, default=0.8, help='Масштаб (по умолчанию 0.8)')

    args = parser.parse_args()

    # Если указаны аргументы командной строки, используем их
    if args.input:
        try:
            if args.batch:
                # Пакетный режим
                input_files = []
                if os.path.isdir(args.input):
                    # Если указана директория, ищем все MusicXML файлы
                    for ext in ['*.xml', '*.mxl', '*.musicxml']:
                        input_files.extend(Path(args.input).glob(ext))
                else:
                    # Если указан файл со списком
                    if args.input.endswith('.txt'):
                        with open(args.input, 'r') as f:
                            input_files = [line.strip() for line in f if line.strip()]
                    else:
                        input_files = [args.input]

                converter = MusicXMLtoPDFConverter(args.lilypond)
                results = converter.batch_convert(input_files, args.output)
                print(f"Конвертация завершена. Успешно: {sum(1 for r in results if r)} файлов")

            else:
                # Одиночный файл
                converter = MusicXMLtoPDFConverter(args.lilypond)
                result = converter.convert_file(
                    args.input,
                    args.output,
                    scale_factor=args.scale
                )
                print(f"Конвертация завершена: {result}")

        except Exception as e:
            print(f"Ошибка: {e}")
            sys.exit(1)
    else:
        main_menu()


if __name__ == "__main__":
    main()