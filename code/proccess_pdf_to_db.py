import os
import re
import cv2
import psycopg2
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path

def get_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres.mbtjtlvdxoznmbjogpmw",
        password="!NLP_MISIS1012",
        host="aws-0-us-west-1.pooler.supabase.com",
        port="5432"
    )

# Статья
def insert_article(cursor, title, language, content):
    cursor.execute(
        "INSERT INTO articles (title, language, content) VALUES (%s, %s, %s) RETURNING id",
        (title, language, content)
    )
    return cursor.fetchone()[0]

# Изображение
def insert_element(cursor, article_id, element_type, path):
    cursor.execute(
        "INSERT INTO elements (article_id, type, path) VALUES (%s, %s, %s) RETURNING id",
        (article_id, element_type, path)
    )
    return cursor.fetchone()[0]

# Фрагмент статьи
def insert_fragment(cursor, article_id, element_id, content):
    cursor.execute(
        "INSERT INTO fragments (article_id, element_id, content) VALUES (%s, %s, %s) RETURNING id",
        (article_id, element_id, content)
    )
    return cursor.fetchone()[0]

def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text(extraction_mode='layout') or "" for page in reader.pages)

def extract_images_from_pdf(pdf_path: Path, output_folder: Path) -> list:
    images_info = []
    images_pil = convert_from_path(str(pdf_path), poppler_path='../poppler-24.08.0/Library/bin')
    os.makedirs(output_folder, exist_ok=True)

    for page_num, img in enumerate(images_pil):
        file_name = f"{pdf_path.stem}_page{page_num + 1}.png"
        img_path = output_folder / file_name
        img.save(img_path, "PNG")
        images_info.append({
            "page": page_num,
            "path": str(img_path),
            "type": "image"
        })
    return images_info

def process_pdf_and_store(pdf_path: Path, output_img_path: Path):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        text = pdf_to_text(pdf_path)
        article_id = insert_article(cursor, pdf_path.stem, "en", text)

        images = extract_images_from_pdf(pdf_path, output_img_path)

        for img in images:
            element_id = insert_element(cursor, article_id, img["type"], img["path"])
            insert_fragment(cursor, article_id, element_id, f"Placeholder for {img['path']}")

        conn.commit()
        print(f"Успешно обработан PDF: {pdf_path.name}")

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при обработке {pdf_path.name}: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    folder = Path("../PDFiles/Files")
    output_images = Path("../PDFiles/Extracted_Images")
    os.makedirs(output_images, exist_ok=True)

    for pdf_file in folder.glob("*.pdf"):
        process_pdf_and_store(pdf_file, output_images)