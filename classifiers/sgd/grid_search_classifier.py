import numpy as np
import classifiers as clss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.pipeline import Pipeline
from classifiers.sgd.transformers import Tokenise, RemoveStopWords, Lemmenise, identity
from classifiers.sgd import dataset

#Best parameters set:
#classifier__alpha: 1e-06
#classifier__penalty: 'l2'
#tfidf__max_df: 0.75
#tfidf__max_features: None
#tfidf__ngram_range: (1, 2)
#tfidf__use_idf: True

STATEMENTS_PATH = 'poltifact.csv'

np.random.seed(1)

pipeline = Pipeline([
    ('tokenise', Tokenise()),
    ('remove_stop_words', RemoveStopWords()),
    ('lemmenise', Lemmenise()),
    ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
    ('classifier', SGDClassifier(n_jobs=-1, max_iter=5))
])

parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__max_features': (None, 5000, 10000, 50000),
    'tfidf__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'classifier__alpha': (0.00001, 0.000001),
    'classifier__penalty': ('l2', 'elasticnet'),
    'classifier__max_iter': (10, 50, 80),
}

# print('Fetching data...')
# X, y = dataset.get_corpus('poltifact.csv')
# X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
#
# print('Fitting data...')
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10)
# grid_search.fit(X_train, y_train)
#
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
# print('Predicting for test set...')
# y_pred = grid_search.predict(X_test)
# print(clsr(y_test, y_pred))


def train_and_classify(train_path, test_path, name=None):
    if name:
        print('====== Training and classifying {} ======'.format(name))

    print('Fetching data...')
    X_train, y_train = dataset.get_corpus(train_path)
    X_test, y_test = dataset.get_corpus(test_path)

    print('Fitting data...')
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10)
    with clss.CodeTimer('Training completed in {:.2f}s'):
        grid_search.fit(X_train, y_train)

    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print('Predicting for test set...')
    y_pred = None
    with clss.CodeTimer('Classification completed in {:.2f}s'):
        y_pred = grid_search.predict(X_test)

    print(clsr(y_test, y_pred))


if __name__ == '__main__':
    train_and_classify(clss.MOVIE_REVIEWS_TRAIN,
                       clss.MOVIE_REVIEWS_TEST,
                       'Movie Reviews')
    train_and_classify(clss.POLITIFACT_TRAIN,
                       clss.POLITIFACT_TEST,
                       'Politifact')
    train_and_classify(clss.ARTICLES_TRAIN,
                       clss.ARTICLES_TEST,
                       'Articles')
