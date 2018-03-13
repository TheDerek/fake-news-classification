import os

from tqdm import tqdm

import transformers as trf
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from collections import namedtuple
from pandas import read_csv


POLTIFACT = 'poltifact.csv'
ARTICLES_TRAIN = 'articles_train.csv'
ARTICLES_TEST ='articles_test.csv'
IMPROVED_ARTICLES = 'improved_articles.csv'
current_path = os.path.dirname(__file__)

Corpus: ([str], [str]) = namedtuple('Corpus', 'data target')
Dataset = namedtuple('Dataset', 'data target labels')
pbar = None

def get_labels(target) -> ([str], [int]):
    labels = list(set(target))
    label_indices = dict(zip(labels, range(len(labels))))
    return labels, [label_indices[l] for l in target]


def get_corpus(file_name: str) -> Corpus:
    path = os.path.join(current_path, '../data', file_name)
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
