import asyncio
import os
import re
import cv2
import psycopg2
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path

from S3_connection import S3Client


def get_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres.mbtjtlvdxoznmbjogpmw",
        password="!NLP_MISIS1012",
        host="aws-0-us-west-1.pooler.supabase.com",
        port="5432"
    )

def get_S3_connection():
    return S3Client(
        access_key="YCAJEI7og7KViaXXyhPIcFgvi",
        secret_key="YCNS3Qu3_9RKoOOXHgu7Y_amzVcLj5qE9S2mK4UA",
        endpoint_url="https://storage.yandexcloud.net",  # для Selectel используйте https://s3.storage.selcloud.ru
        bucket_name="nlpstorage",
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

async def extract_images_from_pdf(pdf_path: Path, output_folder: str, s3conn:S3Client) -> list:
    images_info = []
    images_pil = convert_from_path(str(pdf_path), poppler_path='../poppler-24.08.0/Library/bin')
    os.makedirs(Path(output_folder), exist_ok=True)

    for page_num, img in enumerate(images_pil):
        file_name = f"{pdf_path.stem}_page{page_num + 1}.png"
        img_path = output_folder +"/"+ file_name
        img.save(Path(img_path), "PNG")
        images_info.append({
            "page": page_num,
            "path": str(img_path),
            "type": "image"
        })
        await s3conn.upload_file(img_path, img_path)
    return images_info

async def process_pdf_and_store(file_path: str, output_img_path: str):
    conn = get_connection()
    cursor = conn.cursor()
    s3conn = get_S3_connection()
    local_filepath = "pdf_local_article.pdf"
    await s3conn.get_file(file_path, local_filepath)
    try:
        text = pdf_to_text(Path(local_filepath))
        article_id = insert_article(cursor, Path(file_path).stem, "en", text)

        images = await extract_images_from_pdf(Path(local_filepath), output_img_path, s3conn)

        for img in images:
            element_id = insert_element(cursor, article_id, img["type"], img["path"])
            insert_fragment(cursor, article_id, element_id, f"Placeholder for {img['path']}")

        conn.commit()
        print(f"Успешно обработан PDF: {Path(file_path).name}")

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при обработке {Path(file_path).name}: {e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    """
     folder = Path("../PDFiles/Files")
    output_images = Path("../PDFiles/Extracted_Images")
    os.makedirs(output_images, exist_ok=True)

    for pdf_file in folder.glob("*.pdf"):
        process_pdf_and_store(pdf_file, output_images)
    """
    asyncio.run(process_pdf_and_store("Articles/414.pdf", "Output_Images"))