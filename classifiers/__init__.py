from timeit import default_timer as timer
from pathlib import Path
from sklearn.metrics import classification_report as clsr

import os

from classifiers import dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

MOVIE_REVIEWS_TRAIN = os.path.join(DATA_DIR, 'movie_reviews_train.csv')
MOVIE_REVIEWS_TEST = os.path.join(DATA_DIR, 'movie_reviews_train.csv')

ARTICLES_TRAIN = os.path.join(DATA_DIR, 'articles_train.csv')
ARTICLES_TEST = os.path.join(DATA_DIR, 'articles_test.csv')

POLITIFACT_TRAIN = os.path.join(DATA_DIR, 'politifact_train.csv')
POLITIFACT_TEST = os.path.join(DATA_DIR, 'politifact_test.csv')

WORD_VECTORS = os.path.join(DATA_DIR, 'word-vectors.bin')


class CodeTimer:
    def __init__(self, print_string):
        self.print_string = print_string

    def __enter__(self):
        self.start = timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = timer()
        self.duration = self.end - self.start

        print(self.print_string.format(self.duration))


def show_most_informative_features(vectorizer, clf, n=20):
    """Show the most informative features

    :param vectorizer: The vectorizer that was used to vectorise the dataset.
    :param clf: The classifier that was used to classify the dataset
    :param n: The number of features to show
    """

    # Fetch the features (i.e words) from the vectorizer
    feature_names = vectorizer.get_feature_names()

    # Fetch the weightings of each feature and attach to the corresponding word
    features = sorted(zip(clf.coef_[0], feature_names))

    for class_name in clf.classes_:
        print('\nMost informative {} features:'.format(class_name))
        [print('{}, {}'.format(f[1], f[0])) for f in features[:n]]
        features.reverse()


def train_and_classify(train_path, test_path, pipeline, name=None, show_features=False):
    if name:
        print('====== Training and classifying {} ======'.format(name))

    print('Fetching data...')
    X_train, y_train = dataset.get_corpus(train_path)
    X_test, y_test = dataset.get_corpus(test_path)

    print('Fitting data...')
    with CodeTimer('Training completed in {:.2f}s'):
        pipeline.fit(X_train, y_train)

    print('Predicting for test set...')
    y_pred = None
    with CodeTimer('Classification completed in {:.2f}s'):
        y_pred = pipeline.predict(X_test)

    print(clsr(y_test, y_pred))

    if show_features:
        show_most_informative_features(pipeline.named_steps['tfidf'],
                                       pipeline.named_steps['classifier'])
