import spacy
import re
from collections import defaultdict
from termcolor import colored

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

def get_homonyms(text: str) -> str:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    homonyms = defaultdict(set)
    for token in doc:
        if not token.is_space:
            homonyms[token.text.lower()].add(token.pos_)
    homonyms_filtered = {word: pos_tags for word, pos_tags in homonyms.items() if len(pos_tags) > 1}
    return homonyms_filtered

#если проценты нужны омонимий
def calculate_homonymy_percentage(text: str) -> float:
    lemmatized_text = lemmatization(preprocessing(text))
    homonyms = get_homonyms(lemmatized_text)
    total_words = len(lemmatized_text.split())
    homonym_count = len(homonyms)
    if total_words == 0:
        return 0.0
    percentage = (homonym_count / total_words) * 100
    return percentage


"""
#здесь пример использования
with open('tgt.txt', 'r', encoding='utf-8') as file:#откуда
    text = file.read()
with open('res.txt', 'w', encoding='utf-8') as output_file:#куда
    text = get_homonyms(lemmatization(preprocessing(text)))
    for word, pos_tags in text.items():
        print(colored(f"Слово: {word}", "blue"), colored(f"Омонимий: {len(pos_tags)}", "green"), colored(f"Части речи: {pos_tags}", "cyan"))

    #print(calculate_homonymy_percentage(text))
"""
