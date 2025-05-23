import os
import pandas as pd
import spacy
from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from gensim import corpora, models
from collections import defaultdict
import networkx as nx
import coreferee
import json

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("coreferee")

# Конфиги путей
CSV_FOLDER = r'D:\Code\nlp\PDFFiles\ProcessedCSV'
GRAPH_FOLDER = r'D:\Code\nlp\PDFFiles\GRAPH'
GRAPH_CSV_RELATION = r'D:\Code\nlp\PDFFiles\GRAPH_CSV_RELATION'

# Загрузка модели word2vec
w2v_model_path = r'D:\Code\nlp\word2vec.bin'
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)

CONTROL_VERBS = ["control", "regulate", "inhibit", "activate", "break", "suppress"]
DEPENDENCY_VERBS = ["depend", "require", "need"]
EFFECT_VERBS = ["cause", "lead", "result", "affect", "increase", "decrease"]

def extract_relations(text: str):
    relations = []
    text = nlp(text)
    for sent in text.sents:
        for token in sent:
            # Паттерн 1: nsubj + dobj (Субъект → Действие → Объект)
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.lemma_
                obj = [t for t in token.head.children if t.dep_ == "dobj"]

                if obj:
                    rel_type = "ACTION_OBJ"
                    if verb in CONTROL_VERBS:
                        rel_type = "CONTROL"
                    elif verb in EFFECT_VERBS:
                        rel_type = "EFFECT"

                    relations.append((subject, verb, obj[0].text, rel_type))

            # Паттерн 2: nsubj + prep + pobj (Предложные отношения)
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                preps = [t for t in token.head.children if t.dep_ == "prep"]
                for prep in preps:
                    objs = [t for t in prep.children if t.dep_ == "pobj"]
                    if objs:
                        verb_phrase = f"{token.head.lemma_} {prep.text}"
                        rel_type = "PREPOSITIONAL"
                        if token.head.lemma_ in DEPENDENCY_VERBS:
                            rel_type = "DEPENDENCY"

                        relations.append((token.text, verb_phrase, objs[0].text, rel_type))

            # Паттерн 3: nsubj + xcomp (Глагольное дополнение)
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                xcomps = [t for t in token.head.children if t.dep_ == "xcomp"]
                for xcomp in xcomps:
                    objs = [t for t in xcomp.children if t.dep_ == "dobj"]
                    if objs:
                        relations.append((token.text, f"{token.head.lemma_} {xcomp.lemma_}",
                                          objs[0].text, "XCOMP_EFFECT"))

            # Обработка пассивных конструкций
            if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                agent = [t for t in token.head.children if t.dep_ == "agent"]
                if agent:
                    relations.append((agent[0].text, token.head.lemma_,
                                      token.text, "PASSIVE_REL"))
    return relations

def extract_flat_entities(text):
    """
    Извлекает сущности (существительные и имена собственные) из текста.
    :param text: Текст, из которого извлекаются сущности.
    :return: Список сущностей.
    """
    doc = nlp(text)
    entities = set()
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass", "dobj", "iobj", "pobj", "attr", "advmod", "prep", "prt", "npadvmod"} or token.pos_ == "NOUN":
            span = doc[token.left_edge.i : token.right_edge.i + 1]
            if span.text.strip() and len(span.text) <= 50:
                entities.add(span.text)
    return list(entities)

def get_entity_vector(entity, model):
    """
    Получает векторное представление сущности с использованием модели Word2Vec.
    :param entity: Сущность, для которой нужно получить вектор.
    :param model: Модель Word2Vec для получения векторов слов.
    :return: Векторное представление сущности.
    """
    words = entity.lower().split()
    word_vectors = []
    for w in words:
        if w in model:
            word_vectors.append(model[w])
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size if hasattr(model, 'vector_size') else 300)

def resolve_coref(text: str) -> str:
    """
    Разрешает анафоры в тексте (заменяет "он" на имя).
    :param text: Текст, в котором нужно разрешить анафоры.
    :return: Текст с разрешёнными анафорами.
    """
    doc = nlp(text)
    replacements = []
    for chain in doc._.coref_chains:
        first = list(chain[0])
        rep = doc[first[0]: first[-1] + 1].text
        for mention in chain[1:]:
            idxs = list(mention)
            replacements.append((idxs[0], idxs[-1] + 1, rep))

    tokens = [t.text_with_ws for t in doc]
    for start, end, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
        tokens[start:end] = [rep + " "]
    return "".join(tokens).strip()

def assign_entities_to_topics(entities, df, lda_model, dictionary):
    """
    Назначает сущности к темам на основе модели LDA.
    :param entities: Сущности, для которых нужно назначить тему.
    :param df: DataFrame, содержащий предложения и извлечённые сущности.
    :param lda_model: Модель LDA для выделения тем.
    :param dictionary: Словарь для преобразования текста в Bag-of-Words.
    :return: Словарь, в котором каждой сущности соответствует её тема.
    """
    from collections import defaultdict
    entity_texts = defaultdict(list)
    for desc, ents in zip(df['next_sentences'], df['entities']):
        for ent in ents:
            if ent in entities:
                entity_texts[ent].append(desc.lower())
    entity_docs = {ent: ' '.join(texts).split() for ent, texts in entity_texts.items()}
    entity_topic_assignment = {}
    for ent, tokens in entity_docs.items():
        bow = dictionary.doc2bow(tokens)
        topic_dist = lda_model.get_document_topics(bow)
        if topic_dist:
            best_topic = max(topic_dist, key=lambda x: x[1])[0]
            entity_topic_assignment[ent] = best_topic
        else:
            entity_topic_assignment[ent] = None
    return entity_topic_assignment

def build_graphs_and_save(df, base_name):
    """
    Строит два графа:
    - Один с использованием кластеризации и LDA.
    - Второй с добавлением всех типов связей из extract_relations.
    """
    G_lda_cluster = nx.DiGraph()
    G_all_relations = nx.DiGraph()

    # Строим граф кластеризации + LDA
    all_entities = list({ent for ents in df['entities'] for ent in ents})
    entity_vectors = np.array([get_entity_vector(ent, w2v_model) for ent in all_entities])

    agglo = AgglomerativeClustering(n_clusters=5)
    labels = agglo.fit_predict(entity_vectors)
    clusters = defaultdict(list)
    for entity, label in zip(all_entities, labels):
        clusters[label].append(entity)

    knowledge_graph = {}
    for cluster_id, entities in clusters.items():
        texts = []
        for desc, ents in zip(df['next_sentences'], df['entities']):
            if any(e in entities for e in ents):
                tokens = desc.lower().split()
                texts.append(tokens)
        if not texts:
            continue
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
        entity_topic_map = assign_entities_to_topics(entities, df, lda_model, dictionary)
        topics = lda_model.print_topics(num_words=5)
        topics_dict = {i: [] for i in range(len(topics))}
        for ent, topic_id in entity_topic_map.items():
            if topic_id is not None:
                topics_dict[topic_id].append(ent)
        knowledge_graph[f'Cluster_{cluster_id}'] = {
            'topics': topics,
            'entities_per_topic': topics_dict
        }

    for cluster_id, data in knowledge_graph.items():
        cluster_node = f"Cluster_{cluster_id}"
        G_lda_cluster.add_node(cluster_node, type="cluster")

        for topic_id, topic_words in data['topics']:
            topic_node = f"{cluster_node}_Topic_{topic_id}"
            G_lda_cluster.add_node(topic_node, type="topic")
            G_lda_cluster.add_edge(cluster_node, topic_node)

            entities = data['entities_per_topic'].get(topic_id, [])[:15]
            for entity in entities:
                entity_node = f"{topic_node}_Entity_{entity[:15]}"
                G_lda_cluster.add_node(entity_node, type="entity")
                G_lda_cluster.add_edge(topic_node, entity_node)

    # Строим граф с добавлением всех связей
    for rel in df['relations']:
        if len(rel) > 0 and len(rel[0]) == 4:
            G_all_relations.add_edge(rel[0][0], rel[0][2], relation_type=rel[0][1])

    # Сохраняем графы с разными префиксами
    lda_cluster_json_path = os.path.join(GRAPH_FOLDER, f"{base_name}_lda_cluster.json")
    save_graph_to_json(G_lda_cluster, lda_cluster_json_path)

    all_relations_json_path = os.path.join(GRAPH_FOLDER, f"{base_name}_all_relations.json")
    save_graph_to_json(G_all_relations, all_relations_json_path)

def save_graph_to_json(G, json_path):
    """
    Сохраняет граф в формате JSON.
    :param G: Граф знаний в формате NetworkX.
    :param json_path: Путь для сохранения графа.
    """
    graph_data = nx.node_link_data(G, edges="edges")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4, ensure_ascii=False)
    print(f"Graph saved to {json_path}")

def process_csv_file(csv_path):
    """
    Обрабатывает CSV файл, извлекая сущности и связи, а также разрешая анафоры.
    :param csv_path: Путь к CSV файлу для обработки.
    :return: DataFrame с извлечёнными сущностями и связями.
    """
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df['next_sentences'] = df['next_sentences'].apply(resolve_coref)
    df['entities'] = df['next_sentences'].apply(extract_flat_entities)
    df['relations'] = df['next_sentences'].apply(extract_relations)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    csv_save_path = os.path.join(GRAPH_CSV_RELATION, f"{base_name}_entities_relations.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Entities and relations saved to {csv_save_path}")

    return df

def process_and_build_graph(csv_file_path):
    """
    Обрабатывает CSV файл, строит два графа знаний и сохраняет их в формате JSON.
    :param csv_file_path: Путь к CSV файлу для обработки.
    """
    df = pd.read_csv(csv_file_path)
    df = process_csv_file(csv_file_path)

    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    build_graphs_and_save(df, base_name)


# Вызов работы кода

for file in os.listdir(CSV_FOLDER):
    if file.endswith('.csv'):
        csv_file_path = os.path.join(CSV_FOLDER, file)
        process_and_build_graph(csv_file_path)
