import os
import re
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2docx import Converter
from docx import Document

# PDF → TXT напрямую
def pdf_to_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        print(f"Ошибка {pdf_path.name}: {e}")
        return ""

# PDF → DOCX
def convert_pdf_to_docx(pdf_path: Path, docx_path: Path):
    try:
        cv = Converter(str(pdf_path))
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
    except Exception as e:
        print(f"Ошибка при конвертации {pdf_path.name} в DOCX: {e}")

# DOCX → TXT
def extract_text_from_docx(docx_path: Path) -> str:
    try:
        doc = Document(str(docx_path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"Ошибка чтения DOCX: {docx_path.name}: {e}")
        return ""

# Нормализация абзацев
def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("  ", " ")
    text = re.sub(r"(?<![.\n])\n(?=\w)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def extract_texts_and_normalize(pdf_folder: Path, output_direct: Path, output_from_docx: Path, docx_temp: Path):
    os.makedirs(output_direct, exist_ok=True)
    os.makedirs(output_from_docx, exist_ok=True)
    os.makedirs(docx_temp, exist_ok=True)

    pdf_files = list(pdf_folder.rglob("*.pdf"))
    print(f"Найдено PDF файлов: {len(pdf_files)}")

    for pdf_path in pdf_files:
        name = pdf_path.stem
        print(f"\n Обработка: {name}")

        # 1. PDF → TXT
        text_direct = pdf_to_text(pdf_path)
        text_direct = normalize_text(text_direct)
        if text_direct.strip():
            out_txt_direct = output_direct / f"{name}.txt"
            out_txt_direct.write_text(text_direct, encoding="utf-8")
            print(f"Прямой TXT сохранён: {out_txt_direct.name}")
        else:
            print(f"Нет прямого текста: {pdf_path.name}")

        # 2. PDF → DOCX → TXT
        docx_path = docx_temp / f"{name}.docx"
        convert_pdf_to_docx(pdf_path, docx_path)

        text_docx = extract_text_from_docx(docx_path)
        text_docx = normalize_text(text_docx)
        if text_docx.strip():
            out_txt_docx = output_from_docx / f"{name}.txt"
            out_txt_docx.write_text(text_docx, encoding="utf-8")
            print(f"Текст из DOCX сохранён: {out_txt_docx.name}")
        else:
            print(f"Нет текста через DOCX: {pdf_path.name}")

if __name__ == "__main__":
    pdf_input_folder = Path("../PDFiles/PDFiles/Files") #pdf файлы
    output_text_direct = Path("../PDFiles/PDFiles/Result/TXT_direct") #txt напрямую из пдф
    output_text_from_docx = Path("../PDFiles/PDFiles/Result/TXT_from_docx") #txt через docx
    docx_temp_folder = Path("../PDFiles/PDFiles/Result/DOCX_temp") #Полученные файлы docx
    extract_texts_and_normalize(pdf_input_folder, output_text_direct, output_text_from_docx, docx_temp_folder)