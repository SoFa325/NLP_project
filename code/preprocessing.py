import spacy
import re

def preprocessing(text: str) -> str:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop or token.text == ":" or token.text == "" : #сюда добавлять символы для удаления
            continue
        if re.match(r"([a-zA-Z0-9]|.|,)+", token.text):#сюда добавлять символы для определения
            filtered_tokens.append(token.text.strip())
    return ' '.join(filtered_tokens)

def lemmatization(text: str) -> str:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        lemmatized_tokens.append( token.lemma_)
    return ' '.join(lemmatized_tokens)

#здесь пример использования
with open('tgt.txt', 'r', encoding='utf-8') as file:#откуда
    text = file.read()
with open('res.txt', 'w', encoding='utf-8') as output_file:#куда
    output_file.write(lemmatization(preprocessing(text)))
