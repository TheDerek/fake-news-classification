from timeit import default_timer as timer

import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

MOVIE_REVIEWS_TRAIN = os.path.join(DATA_DIR, 'movie_reviews_train.csv')
MOVIE_REVIEWS_TEST = os.path.join(DATA_DIR, 'movie_reviews_train.csv')

ARTICLES_TRAIN = os.path.join(DATA_DIR, 'articles_train.csv')
ARTICLES_TEST = os.path.join(DATA_DIR, 'articles_test.csv')

POLITIFACT_TRAIN = os.path.join(DATA_DIR, 'politifact_train.csv')
POLITIFACT_TEST = os.path.join(DATA_DIR, 'politifact_test.csv')

WORD_VECTORS = os.path.join(DATA_DIR, 'GoogleNews-vectors-negative300-SLIM.bin')


class CodeTimer:
    def __init__(self, print_string):
        self.print_string = print_string

    def __enter__(self):
        self.start = timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = timer()
        self.duration = self.end - self.start

        print(self.print_string.format(self.duration))
