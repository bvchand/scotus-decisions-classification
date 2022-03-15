from scotus_data_extraction import extract_data

import pickle
import nltk
import re
import numpy as np
from tqdm import tqdm
from os import path
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


def data_preprocess(scotus_text):
    tagged_scotus_text = []

    for i, list_of_words in tqdm(enumerate(scotus_text)):
        text_without_punct = re.sub(r'[^\w\s]', ' ', str(list_of_words))
        tagged_scotus_text.append(TaggedDocument(words=word_tokenize(text_without_punct.lower()), tags=[str(i)]))

    # pickle.dump(tagged_scotus_text, open(
    #     "/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/tagged_scotus_text_sample.pkl", "wb"))
    return tagged_scotus_text


def doc2vec(tagged_scotus_text, scotus_text):
    model = Doc2Vec(vector_size=1000, window=10, min_count=5, workers=4, alpha=0.025, min_alpha=0.005)
    model.build_vocab(tagged_scotus_text, keep_raw_vocab=True)
    model.train(tagged_scotus_text, total_examples=len(scotus_text), epochs=25)
    model.save("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/d2v_sample.model")


def text_to_doc2vec(scotus_text):
    model_path = "/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/d2v_sample.model"

    if not path.exists(model_path):
        tagged_scotus_text = data_preprocess(scotus_text)
        doc2vec(tagged_scotus_text, scotus_text)
        d2v_model = Doc2Vec.load(model_path)

    else:
        d2v_model = Doc2Vec.load(model_path)

    return d2v_model


def preprocess_doc2vec(scotus_text):
    d2v_model = text_to_doc2vec(scotus_text)
    X = np.array([d2v_model.docvecs[str(i)] for i in range(1)])
    return X
