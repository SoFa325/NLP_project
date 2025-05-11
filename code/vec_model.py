from gensim.models import Word2Vec, KeyedVectors

def train_word2vec(documents):
    model = Word2Vec(
        documents,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=50
    )
    return model

def save_word2vec_model(model, filepath_bin):
    model.wv.save_word2vec_format(filepath_bin, binary=True)
    print(f"Модель сохранена в {filepath_bin}")

