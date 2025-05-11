import pdf_extr
from pathlib import Path

pdf_input_folder = Path(r"../PDFiles/Files")
output_html_folder = Path(r"../PDFiles/HTML")
image_output_folder = Path(r"../PDFiles/Cropped_Images2")   # Директория для сохранения изображений
image_pdf_output_folder = Path(r"../PDFiles/PDF_to_IMG")
csv_output_folder = Path(r"../PDFiles/CSV")

pdf_extr.extract_texts_from_folder(
    pdf_input_folder,
    output_html_folder,
    image_output_folder,
    image_pdf_output_folder,
    csv_output_folder
)