import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report as clsr

import classifiers as clss
import numpy as np

from classifiers import dataset


class DeepNeuralNetworkClassifier(BaseEstimator):
    def __init__(self):
        self._feature_columns = None
        self._classifier = None
        self._labels = {
            0: 'FAKE',
            1: 'REAL'
        }

    def fit(self, X, y):
        self._feature_columns = [tf.feature_column.numeric_column("x",
                                                                  shape=[X.shape[1]])]
        self._classifier = tf.estimator.DNNClassifier(
            feature_columns=self._feature_columns, hidden_units=[100, 50, 20],
            n_classes=2)

        #y = [np.array([[1.] if i == l else [0.] for l in self._labels]) for i in y]
        y = [0. if i == self._labels[0] else 1. for i in y]
        y = np.array(y)

        # Define the training inputs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X.toarray()},
            y=y,
            num_epochs=None,
            shuffle=True)

        # Train model.
        print('Training model...')
        self._classifier.train(input_fn=train_input_fn, steps=2000)

    def predict(self, X):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X.toarray()},
            num_epochs=1,
            shuffle=False)
        
        predictions = list(self._classifier.predict(input_fn=test_input_fn))
        predicted_classes = [p["classes"] for p in predictions]
        return [self._labels[int(c[0])] for c in predicted_classes]


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, stop_words='english')),
    ('classifier', DeepNeuralNetworkClassifier())
])


def main():
    print('Fetching data...')
    X_train, y_train = dataset.get_corpus(clss.POLITIFACT_TRAIN)
    X_test, y_test = dataset.get_corpus(clss.POLITIFACT_TEST)

    print('Setting up model...')

    # Train model.
    print('Training model...')
    pipeline.fit(X_train, y_train)

    print('Testing model...')
    y_pred = pipeline.predict(X_test)
    print(clsr(y_test, y_pred))


if __name__ == "__main__":
    # clss.train_and_classify(clss.MOVIE_REVIEWS_TRAIN, clss.MOVIE_REVIEWS_TEST, pipeline,
    #                         'Movie Reviews')
    # clss.train_and_classify(clss.POLITIFACT_TRAIN, clss.POLITIFACT_TEST, pipeline,
    #                         'Politifact')
    clss.train_and_classify(clss.ARTICLES_TRAIN, clss.ARTICLES_TEST, pipeline,
                            'Articles')
