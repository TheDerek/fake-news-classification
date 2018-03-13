import gensim
import classifiers as cls
import nltk
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from nltk.corpus import stopwords as sw


class Transformer:
    def __init__(self):
        self._tokenizer = nltk.RegexpTokenizer(r'\b[^\d\W]+\b')
        self._lemmeniser = nltk.WordNetLemmatizer()
        self._stopwords = sw.words('english')
        self._model = KeyedVectors.load_word2vec_format(cls.WORD_VECTORS, binary=True)

    def get_embeddings(self, document):
        return np.array([self._model.wv[token] for token in self.get_tokens(document)])

    def can_use_token(self, token):
        return token not in self._stopwords and token in self._model.vocab

    def get_tokens(self, document):
        return (token for token in self._tokenizer.tokenize(document) if
                self.can_use_token(token))

    def get_corpus(self, filepath):
        df = pd.read_csv(filepath)

        # Fetch the labels as numbers
        labels = list(set(df.label))
        label_indices = dict(zip(labels, range(len(labels))))
        y = np.array([label_indices[l] for l in df.label])

        X = [self.get_embeddings(document) for document in df.text]

        return X, y
