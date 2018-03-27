import os
import numpy as np

from classifiers import transformers as trf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from collections import namedtuple
from pandas import read_csv
from tqdm import tqdm


Corpus = namedtuple('Corpus', 'data target')
Dataset = namedtuple('Dataset', 'data target labels')
pbar = None


def get_labels(target) -> ([str], [int]):
    labels = list(set(target))
    label_indices = dict(zip(labels, range(len(labels))))
    return labels, [label_indices[l] for l in target]


def get_corpus(path: str) -> Corpus:
    data = read_csv(path)
    data = data.sample(frac=1)
    return Corpus(data.text.tolist(), data.label.tolist())


def corpus_to_data(corpus: Corpus) -> Dataset:
    """Turn a corpus of text into a tuple of numpy arrays suitable for machine
    learning."""

    pipeline = Pipeline([
        ('tokenise', trf.Tokenise()),
        ('remove_stop_words', trf.RemoveStopWords()),
        ('lemmenise', trf.Lemmenise()),
        ('tfidf', TfidfVectorizer(tokenizer=trf.identity, lowercase=False))
    ])

    X = pipeline.fit_transform(corpus.data)
    labels, y = get_labels(corpus.target)

    return Dataset(X, np.array(y), labels)


def get_data(file_name: str) -> Dataset:
    return corpus_to_data(get_corpus(file_name))
