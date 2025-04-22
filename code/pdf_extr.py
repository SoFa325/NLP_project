import warnings
import re
import os
from pathlib import Path
import cv2
from pypdf import PdfReader
from pdf2image import convert_from_path
import pandas as pd

class PdfExtraction:
    def __init__(self, poppler_path: Path):
        self.poppler_path = poppler_path

    def pdf_to_text(self, pdf_path: Path) -> list:
        try:
            reader = PdfReader(str(pdf_path))
            text = [page.extract_text(extraction_mode='layout') or "" for page in reader.pages]
            return text
        except Exception as e:
            print(f"Ошибка при извлечении текста из {pdf_path.name}: {e}")
            return []

    def extract_images_from_pdf(self, pdf_path: Path, image_output_folder: Path, image_pdf_output_folder: Path) -> list:
        images = []
        try:
            images_pil = convert_from_path(
                str(pdf_path),
                poppler_path=str(self.poppler_path)
            )

            doc_name = pdf_path.stem
            os.makedirs(image_output_folder, exist_ok=True)
            os.makedirs(image_pdf_output_folder, exist_ok=True)

            for page_num, img in enumerate(images_pil):
                # Сохраним страницу (необязательно, можно убрать)
                page_png = image_pdf_output_folder / f"{doc_name}_page{page_num+1}.png"
                img.save(page_png, 'PNG')

                img_cv = cv2.imread(str(page_png))
                if img_cv is None:
                    print(f"Не удалось прочитать {page_png}")
                    continue

                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                image_counter = 1
                for contour in contours:
                    if cv2.contourArea(contour) > 35000:
                        x, y, w, h = cv2.boundingRect(contour)
                        cropped = img_cv[y:y+h, x:x+w]

                        cropped_filename = image_output_folder / f"{doc_name}_cropped_page{page_num+1}_{image_counter}.png"
                        cv2.imwrite(str(cropped_filename), cropped)

                        images.append({
                            "page": page_num,
                            "image_filename": cropped_filename
                        })
                        image_counter += 1
        except Exception as e:
            print(f"Ошибка при извлечении изображений из {pdf_path.name}: {e}")

        return images

    def extract_figures_and_descriptions(self, pdf_path: Path, output_html: Path, image_output_folder: Path, image_pdf_output_folder: Path, csv_output_folder: Path) -> list:
        text_pages = self.pdf_to_text(pdf_path)
        if not text_pages:
            print(f"Нет текста в {pdf_path.name}")
            return []

        images = self.extract_images_from_pdf(pdf_path, image_output_folder, image_pdf_output_folder)
        figures_and_descriptions = []
        image_idx = 0

        for page_num, page_text in enumerate(text_pages):
            cleaned_text = page_text

            captions = [m.group(0)
                        for m in re.finditer(r'\b(Fig|Figure)\s*\.?\s*\d+[A-Za-z]*[\)\.:]*\b', cleaned_text)]
            
            captions = [self.clean_figure_caption(caption) for caption in captions]

            caption_for_page = captions[0] if len(captions) == 1 else None

            local_count = 0
            while image_idx < len(images) and images[image_idx]["page"] == page_num:
                caption = caption_for_page if caption_for_page else (captions[local_count] if local_count < len(captions) else "Figure Not Found")
                figures_and_descriptions.append({
                    'file': pdf_path.name,
                    'figure': caption,
                    'image_file': images[image_idx]["image_filename"],
                    'next_sentences': cleaned_text
                })
                image_idx += 1
                local_count += 1

        self._save_to_html(output_html, figures_and_descriptions)
        self._save_to_csv(csv_output_folder, pdf_path, figures_and_descriptions)
        return figures_and_descriptions

    def clean_figure_caption(self, caption: str) -> str:
        caption = re.sub(r'\.\s+', '.', caption)
        caption = re.sub(r'\bFig\.\s*(\d+)\b', r'Figure \1', caption)
        return caption

    def _save_to_html(self, output_html: Path, figures_and_descriptions: list):
        html = """
<html>
<head>
  <title>Figures and Descriptions</title>
  <style>
    table {border-collapse: collapse; width: 100%;}
    th, td {border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top;}
    th {background: #eee;}
  </style>
</head>
<body>
  <h1>Extracted Figures and Descriptions</h1>
  <table>
    <tr><th>File</th><th>Figure</th><th>Image</th><th>Page Text</th></tr>
"""
        for item in figures_and_descriptions:
            safe_text = (item['next_sentences']
                         .replace('&', '&amp;')
                         .replace('<', '&lt;')
                         .replace('>', '&gt;')
                         .replace('\n', '<br>'))
            html += f"""
    <tr>
      <td>{item['file']}</td>
      <td>{item['figure']}</td>
      <td>{item['image_file']}</td>
      <td>{safe_text}</td>
    </tr>
"""
        html += """
  </table>
</body>
</html>
"""
        try:
            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Results saved to {output_html}")
        except Exception as e:
            print(f"Ошибка при сохранении HTML: {e}")

    def _save_to_csv(self, csv_output_folder: Path, pdf_path: Path, figures_and_descriptions: list):
        try:
            csv_filename = csv_output_folder / f"{pdf_path.stem}.csv"
            os.makedirs(csv_output_folder, exist_ok=True)

            df = pd.DataFrame(figures_and_descriptions)
            df.to_csv(csv_filename, index=False, encoding='utf-8')

            print(f"Results saved to {csv_filename}")
        except Exception as e:
            print(f"Ошибка при сохранении CSV: {e}")

def extract_texts_from_folder(pdf_folder: Path, output_folder: Path, image_output_folder: Path, image_pdf_output_folder: Path, csv_output_folder: Path, poppler_path: Path):
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = list(pdf_folder.rglob("*.pdf"))
    print(f"Найдены {len(pdf_files)} PDF файлы в {pdf_folder}")

    extractor = PdfExtraction(poppler_path=poppler_path)

    for pdf_path in pdf_files:
        output_html = output_folder / f"{pdf_path.stem}_figures.html"
        extractor.extract_figures_and_descriptions(pdf_path, output_html, image_output_folder, image_pdf_output_folder, csv_output_folder)

# Динамическое определение пути родительской директории
parent_path = Path(__file__).parent  # Получаем путь к родительской директории

# # Создание всех подпапок от родительской директории
# poppler_path = parent_path / "poppler-24.08.0" / "Library" / "bin"
# pdf_input_folder = parent_path / "PDFFiles" / "Files"
# output_html_folder = parent_path / "PDFFiles" / "HTML"
# image_output_folder = parent_path / "PDFFiles" / "Cropped_Images2"
# image_pdf_output_folder = parent_path / "PDFFiles" / "PDF_to_IMG"
# csv_output_folder = parent_path / "PDFFiles" / "CSV"


poppler_path = Path(r"D:\Code\poppler-24.08.0\Library\bin")
pdf_input_folder = Path(r"D:\Code\nlp\PDFFiles\Files")
output_html_folder = Path(r"D:\Code\nlp\PDFFiles\HTML")
image_output_folder = Path(r"D:\Code\nlp\PDFFiles\Cropped_Images2")
image_pdf_output_folder = Path(r"D:\Code\nlp\PDFFiles\PDF_to_IMG")
csv_output_folder = Path(r"D:\Code\nlp\PDFFiles\CSV")


# Запуск процесса обработки
extract_texts_from_folder(pdf_input_folder, output_html_folder, image_output_folder, image_pdf_output_folder, csv_output_folder, poppler_path)
