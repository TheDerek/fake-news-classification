import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.pipeline import Pipeline

import classifiers as clss
from classifiers import dataset
from classifiers.transformers import Tokenise, RemoveStopWords, Lemmenise, identity

# Best parameters set:
# classifier__alpha: 1e-06
# classifier__penalty: 'l2'
# tfidf__max_df: 0.75
# tfidf__max_features: None
# tfidf__ngram_range: (1, 2)
# tfidf__use_idf: True

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

if __name__ == '__main__':
    clss.train_and_classify(clss.MOVIE_REVIEWS_TRAIN, clss.MOVIE_REVIEWS_TEST, pipeline,
                            'Movie Reviews', True)
    clss.train_and_classify(clss.POLITIFACT_TRAIN, clss.POLITIFACT_TEST, pipeline,
                            'Politifact', True)
    clss.train_and_classify(clss.ARTICLES_TRAIN, clss.ARTICLES_TEST, pipeline,
                            'Articles', True)
