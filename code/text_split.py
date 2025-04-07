import re

def split_paragraphs_after_introduction(file_path: str) -> list:
    """
    Читает текстовый файл, находит раздел 'introduction', обрезает всё до него,
    затем делит оставшийся текст на абзацы.
    
    :param file_path: Путь к текстовому файлу
    :return: Список абзацев (без пустых строк)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Ищем позицию 'introduction' в любом регистре
    start_index = text.lower().find('introduction')
    
    # Обрезаем текст
    trimmed_text = text[start_index:] if start_index != -1 else text
    
    # Делим на абзацы по одному или нескольким переводам строк с пробелами
    paragraphs = re.split(r'\n ', trimmed_text)
    
    # Очищаем абзацы от пробелов и удаляем пустые
    return [p.strip() for p in paragraphs if p.strip()]
