import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import preprocessing
import vec_model

KEYWORDS = [
    'figure', 'fig', 'diagram', 'schematic', 'layout', 'overview',
    'shows', 'illustrates', 'depicts', 'demonstrates', 'represents',
    'presents', 'indicates', 'displays', 'visualizes',
]

def keyword_density(text):
    words = text.lower().split()
    count = sum(word in KEYWORDS for word in words)
    return count / (len(words) + 1e-5)

def sentence_vector(sentence):
    words = sentence.lower().split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def clean_figure_caption(caption: str) -> str:
    caption = re.sub(r'\bFig\.\s*(\d+)\b', r'Figure \1', caption)
    return caption.strip()

def pdf_to_text(pdf_path: Path) -> list:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text(extraction_mode='layout') or "" for page in reader.pages]

def is_significantly_different(new_rect, existing_rects, threshold=0.85):
    x1, y1, w1, h1 = new_rect
    keep_boxes = []

    for (x2, y2, w2, h2) in existing_rects:
        if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
            print("New region is fully inside an existing box — skipping")
            return False

        if x2 >= x1 and y2 >= y1 and x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1:
            print("🔄 Replacing smaller existing box with larger one")
            continue

        dx = min(x1 + w1, x2 + w2) - max(x1, x2)
        dy = min(y1 + h1, y2 + h2) - max(y1, y2)
        if dx > 0 and dy > 0:
            intersection = dx * dy
            union = w1 * h1 + w2 * h2 - intersection
            ratio = intersection / union
            print(f"Overlap ratio: {ratio:.2f} (Threshold: {threshold})")
            if ratio > threshold:
                print("Skipped as duplicate region.")
                return False

        keep_boxes.append((x2, y2, w2, h2))

    existing_rects[:] = keep_boxes
    return True

def boxes_are_close(b1, b2, gap=20):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    l1, r1, t1, b1 = x1, x1 + w1, y1, y1 + h1
    l2, r2, t2, b2 = x2, x2 + w2, y2, y2 + h2

    horizontal_close = abs(l1 - r2) < gap or abs(l2 - r1) < gap
    vertical_close = abs(t1 - b2) < gap or abs(t2 - b1) < gap

    overlap_x = not (r1 < l2 or r2 < l1)
    overlap_y = not (b1 < t2 or b2 < t1)

    return (overlap_x and vertical_close) or (overlap_y and horizontal_close)


def merge_close_boxes(boxes, gap):
    def boxes_are_close(b1, b2, gap):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (
                x1 + w1 + gap < x2 or x2 + w2 + gap < x1 or
                y1 + h1 + gap < y2 or y2 + h2 + gap < y1
        )

    merged = boxes[:]
    changed = True

    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(merged)

        for i, box1 in enumerate(merged):
            if used[i]:
                continue
            group = [box1]
            used[i] = True
            for j, box2 in enumerate(merged):
                if i != j and not used[j] and boxes_are_close(box1, box2, gap):
                    group.append(box2)
                    used[j] = True
            if len(group) > 1:
                changed = True
            xs = [x for x, y, w, h in group]
            ys = [y for x, y, w, h in group]
            x2s = [x + w for x, y, w, h in group]
            y2s = [y + h for x, y, w, h in group]
            new_boxes.append((min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)))

        merged = new_boxes

    return merged


def extract_images_from_pdf(pdf_path: Path, image_output_folder: Path, image_pdf_output_folder: Path) -> list:
    images = []
    images_pil = convert_from_path(str(pdf_path), poppler_path=r'../poppler-24.08.0/Library/bin')
    doc_name = pdf_path.stem
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(image_pdf_output_folder, exist_ok=True)

    for page_num, img in enumerate(images_pil):
        page_path = image_pdf_output_folder / f"{doc_name}_page{page_num + 1}.png"
        img.save(page_path, 'PNG')

        img_cv = cv2.imread(str(page_path))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        thresh_path = image_pdf_output_folder / f"{doc_name}_thresh_page{page_num + 1}.png"
        cv2.imwrite(str(thresh_path), thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 35000]
        merged_rects = merge_close_boxes(rects, gap=180)

        image_counter = 1
        existing_boxes = []
        for (x, y, w, h) in merged_rects:
            if not is_significantly_different((x, y, w, h), existing_boxes, threshold=0.85):
                continue  # Пропускаем дубликат
            existing_boxes.append((x, y, w, h))  # Добавляем, если уникален
            cropped = img_cv[y:y + h, x:x + w]
            cropped_filename = image_output_folder / f"{doc_name}_cropped_page{page_num + 1}_{image_counter}.png"
            cv2.imwrite(str(cropped_filename), cropped)
            images.append({"page": page_num, "image_filename": cropped_filename})
            image_counter += 1

    return images

def extract_figures_and_descriptions(pdf_path: Path, output_html: Path, image_output_folder: Path, image_pdf_output_folder: Path, csv_output_folder: Path):
    text_pages = pdf_to_text(pdf_path)
    if not text_pages:
        print(f"Нет текста в {pdf_path.name}")
        return

    images = extract_images_from_pdf(pdf_path, image_output_folder, image_pdf_output_folder)
    figures_and_descriptions = []
    image_idx = 0

    used_figures = set()

    for page_num, page_text in enumerate(text_pages):

        cleaned_text = page_text.replace('\n', ' ')
        cleaned_text = re.sub(r'(Figure\s+\d+[A-Za-z]*)\.\s+(?=[A-Z])', r'\1 ', cleaned_text)
        sentences = re.split(r'(?<=[.!?]) +', cleaned_text)

        captions = [clean_figure_caption(m.group(0)) for m in re.finditer(r'\b(Fig(?:ure)?)\.?\s*\(?\d+[A-Za-z]*\)?[\.:]*\b', cleaned_text)]

        local_count = 0

        while image_idx < len(images) and images[image_idx]["page"] == page_num:
            caption = captions[local_count] if local_count < len(captions) else "Figure Not Found"
            description = ""

            caption_number = re.findall(r'\d+', caption)
            if caption_number:
                figure_id = caption_number[0]
                if (figure_id, page_num) in used_figures:
                    print(f"⚠️  Figure {figure_id} already processed on page {page_num+1} — skipping duplicate")
                    image_idx += 1
                    local_count += 1
                    continue
                else:
                    used_figures.add((figure_id, page_num))

            found = False

            if caption_number:
                figure_id = caption_number[0]
                for i, sent in enumerate(sentences):
                    if figure_id in sent and ("Figure" in sent or "Fig" in sent):
                        base_vec = sentence_vector(sent)
                        base_sentence = sent
                        description += sent + " "
                        found = True
                        break

                if found:
                    for j, sent in enumerate(sentences):
                        if sent == base_sentence:
                            continue

                        normalized = sent.lower()
                        has_figure_ref = re.search(rf'(fig(?:ure)?\.?\s*\(?({figure_id})[a-zA-Z]?\)?[\.:]?)', normalized)
                        vec = sentence_vector(sent)
                        sim = cosine_similarity(base_vec.reshape(1, -1), vec.reshape(1, -1))[0][0]

                        reason = []
                        if has_figure_ref:
                            reason.append("by fig ref")
                        if 0.75 < sim < 1.0:
                            reason.append("by cosine")
                        if keyword_density(sent) > 0.05:
                            reason.append("by keyword")

                        # print(f"SIM: {sim:.3f} | SENT: {sent.strip()} | REASON: {', '.join(reason) if reason else '—'}")

                        if reason:
                            description += sent + " "

            if not description:
                description = cleaned_text

            unique_sentences = []
            seen = set()
            for s in re.split(r'(?<=[.!?])\s+', description.strip()):
                s_clean = s.strip()
                if s_clean and s_clean not in seen:
                    unique_sentences.append(s_clean)
                    seen.add(s_clean)
            description = ' '.join(unique_sentences)

            figures_and_descriptions.append({
                'file': pdf_path.name,
                'figure': caption,
                'image_file': images[image_idx]["image_filename"],
                'next_sentences': description.strip()
                #TODO ???
                #'graph':
            })

            image_idx += 1
            local_count += 1

    # HTML вывод
    html = """
<html>
<head>
  <title>Figures and Descriptions</title>
  <style>
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #eee; }
  </style>
</head>
<body>
  <h1>Extracted Figures and Descriptions</h1>
  <table>
    <tr><th>File</th><th>Figure</th><th>Image</th><th>Description</th></tr>
"""
    for item in figures_and_descriptions:
        safe_text = item['next_sentences'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
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
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Results saved to {output_html}")

    # CSV вывод
    csv_filename = csv_output_folder / f"{pdf_path.stem}.csv"
    os.makedirs(csv_output_folder, exist_ok=True)
    pd.DataFrame(figures_and_descriptions).to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"Results saved to {csv_filename}")







"""
def load_documents(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = preprocessing.lemmatization(preprocessing.preprocessing(text)).split()
                #tokens = text
                documents.append(tokens)
                filenames.append(filename)
    return documents, filenames
"""
def load_raw_documents(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = preprocessing.lemmatization(preprocessing.preprocessing(text)).split()#text.lower().split()
                documents.append(tokens)
                filenames.append(filename)
    return documents, filenames

def load_word2vec_model(path):
    global model
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    print("Word2Vec model loaded")

def extract_texts_from_folder(pdf_folder: Path, output_folder: Path, image_output_folder: Path, image_pdf_output_folder: Path, csv_output_folder:Path):
    # Обучение модели
    # Загружаем необработанные документы
    docs, _ = load_raw_documents(r"../PDFiles/Result/TXT_from_docx")

    # Обучаем модель
    w2v_model = vec_model.train_word2vec(docs)

    # Сохраняем
    vec_model.save_word2vec_model(w2v_model, "word2vec.bin")

    # Загружаем перед запуском анализа PDF
    load_word2vec_model("word2vec.bin")

    os.makedirs(output_folder, exist_ok=True)
    pdf_files = list(pdf_folder.rglob("*.pdf"))
    print(f"Найдены {len(pdf_files)} PDF файлы в {pdf_folder}")

    for pdf_path in pdf_files:
        output_html = output_folder / f"{pdf_path.stem}_figures.html"
        extract_figures_and_descriptions(pdf_path, output_html, image_output_folder, image_pdf_output_folder, csv_output_folder)
