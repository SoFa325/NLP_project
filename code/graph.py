import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')# "en_core_web_sm" для скорости # en_core_web_trf для точности

CONTROL_VERBS = ["control", "regulate", "inhibit", "activate", "break", "suppress"]
DEPENDENCY_VERBS = ["depend", "require", "need"]
EFFECT_VERBS = ["cause", "lead", "result", "affect", "increase", "decrease"]

def extract_relations(text:str):
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
    #print("{:<20} {:<20} {:<20} {:<15}".format("Subject", "Relation", "Object", "Type"))
    #print("-" * 80)
    #for rel in relations:
        #print("{:<20} {:<20} {:<20} {:<15}".format(rel[0], rel[1], rel[2], rel[3]))

    # Визуализация дерева зависимостей для первого предложения
    #print("\nВизуализация дерева зависимостей:")
    #displacy.render(list(text.sents)[0], style="dep", options={"compact": True, "distance": 120})
    return relations
