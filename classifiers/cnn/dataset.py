import gensim
import classifiers as cls
import nltk
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from nltk.corpus import stopwords as sw

from classifiers import cnn


class Dataset:
    def __init__(self):
        self._tokenizer = nltk.RegexpTokenizer(r'\b[^\d\W]+\b')
        self._lemmeniser = nltk.WordNetLemmatizer()
        self._stopwords = sw.words('english')
        self._model = KeyedVectors.load_word2vec_format(cls.WORD_VECTORS, binary=True)

    def get_embeddings(self, document, max_size=-1, flatten=False):
        embeddings = np.array([self._model.wv[token]
                               for token in self.get_tokens(document)])

        if max_size > 0:
            embeddings = self.pad_to_size(embeddings, max_size)

        if flatten:
            return embeddings.flatten()

        return embeddings

    def pad_to_size(self, array, max_size):
        result = np.zeros((max_size, 300), dtype=np.float32)

        if len(array) > 0:
            result[:min(array.shape[0], max_size), :array.shape[1]] \
                = array[:min(array.shape[0], max_size), :]

        return result

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

        # X = np.empty((len(df), cnn.TEXT_IMAGE_SIZE))
        # for document in df.text:
        #     np.append(X, self.get_embeddings(document, cnn.TEXT_IMAGE_HEIGHT, True))

        X = np.array([np.array(self.get_embeddings(d, cnn.TEXT_IMAGE_HEIGHT, True))
                      for d in df.text])

        return X, y
