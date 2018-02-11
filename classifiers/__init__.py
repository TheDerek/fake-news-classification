import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

MOVIE_REVIEWS_TRAIN = os.path.join(DATA_DIR, 'movie_reviews_train.csv')
MOVIE_REVIEWS_TEST = os.path.join(DATA_DIR, 'movie_reviews_train.csv')

ARTICLES_TRAIN = os.path.join(DATA_DIR, 'articles_train.csv')
ARTICLES_TEST = os.path.join(DATA_DIR, 'articles_test.csv')

POLITIFACT_TRAIN = os.path.join(DATA_DIR, 'politifact_train.csv')
POLITIFACT_TEST = os.path.join(DATA_DIR, 'politifact_test.csv')