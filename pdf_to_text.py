#ЛОКАЛЬНОЕ ЧТЕНИЕ PDF
import os
from pathlib import Path
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        print(f"Ошибка {pdf_path.name}: {e}")
        return ""

def extract_texts_from_folder(pdf_folder: Path, output_folder: Path):
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = list(pdf_folder.rglob("*.pdf"))
    print(f"Найдены {len(pdf_files)} PDF файлы в {pdf_folder}")

    for pdf_path in pdf_files:
        text = pdf_to_text(pdf_path)
        if text.strip():
            out_path = output_folder / f"{pdf_path.stem}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f" Сохранено: {out_path}")
        else:
            print(f" Нет текста из извлеченного: {pdf_path.name}")

if __name__ == "__main__":
    # Файлы PDF (не архив)
    pdf_input_folder = Path("./PDFiles/Files/")
    # Сохраненные файлы (.txt)
    output_text_folder = Path("./PDFiles/Result/")
    extract_texts_from_folder(pdf_input_folder, output_text_folder)



"""ПРИНИМАЕТ PDF-файл через HTTP-запрос, извлекает из него текст и возвращает его в формате JSON
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile
import uvicorn

app = FastAPI(title="PDF API", description="Принимает PDF и возвращает извлечённый текст")

def pdf_to_text(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении текста: {e}")

@app.post("/pdf-to-text")
async def convert_pdf_to_text(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Загружаемый файл должен быть PDF")
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        text = pdf_to_text(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""