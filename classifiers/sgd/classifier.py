import numpy as np
import classifiers as clss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
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

np.random.seed(1)

optimised_pipeline = Pipeline([
    ('tokenise', Tokenise()),
    ('remove_stop_words', RemoveStopWords()),
    ('lemmenise', Lemmenise()),
    ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False, use_idf=True,
                              ngram_range=(1, 2), max_features=None, max_df=0.75,
                              min_df=0.04)),
    ('classifier', SGDClassifier(n_jobs=-1, max_iter=5, tol=None, alpha=1e-06,
                                 penalty='l2'))
])

pipeline = Pipeline([
    ('tokenise', Tokenise()),
    ('remove_stop_words', RemoveStopWords()),
    ('lemmenise', Lemmenise()),
    ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
    ('classifier', SGDClassifier(n_jobs=-1, max_iter=5))
])


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


def train_and_classify(train_path, test_path, name=None):
    if name:
        print('====== Training and classifying {} ======'.format(name))

    print('Fetching data...')
    X_train, y_train = dataset.get_corpus(train_path)
    X_test, y_test = dataset.get_corpus(test_path)

    print('Fitting data...')
    with clss.CodeTimer('Training completed in {:.2f}s'):
        pipeline.fit(X_train, y_train)

    print('Predicting for test set...')
    y_pred = None
    with clss.CodeTimer('Classification completed in {:.2f}s'):
        y_pred = pipeline.predict(X_test)

    print(clsr(y_test, y_pred))

    show_most_informative_features(pipeline.named_steps['tfidf'],
                                   pipeline.named_steps['classifier'])


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
