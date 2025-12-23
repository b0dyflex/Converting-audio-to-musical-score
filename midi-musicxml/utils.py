import os


def validate_file_path(file_path):
    """
    Проверяет существование файла
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    return True


def ensure_directory_exists(directory_path):
    """
    Создает директорию, если она не существует
    """
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path


def get_midi_extensions():
    """
    Возвращает список поддерживаемых расширений MIDI файлов
    """
    return ['.mid', '.midi', '.mid', '.MID', '.MIDI']


def is_valid_midi_extension(file_path, midi_extensions):
    """
    Проверяет, имеет ли файл расширение MIDI
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in midi_extensions


def format_file_size(size_in_bytes):
    """
    Форматирует размер файла в читаемый вид
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"


def get_output_filename(input_path, output_path=None):
    """
    Генерирует имя выходного файла
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.musicxml"
    elif not output_path.lower().endswith(('.musicxml', '.xml', '.mxl')):
        output_path = f"{output_path}.musicxml"

    return output_path