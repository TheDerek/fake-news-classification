#!/usr/bin/python3

from collections import Counter
from random import shuffle
from tqdm import tqdm
from classifiers.naive_bayes import dataset
from classifiers import CodeTimer

import classifiers as clss
import nltk


def most_common_words(dt, limit=None):
    word_count = Counter()
    for article, classification in tqdm(dt):
        word_count += Counter(article['text_words'])

    most_common = word_count.most_common(limit)
    return [i[0] for i in most_common]


def document_features(document, common_words):
    article_words = set(document['text_words'])
    features = {}
    for word in common_words:
        features['contains({})'.format(word)] = (word in article_words)

    return features


def train_and_classify(train_path, test_path, name=None):
    if name:
        print('====== Training and classifying {} ======'.format(name))

    print('Loading dataset...')
    train_set = dataset.load(train_path)
    print('\tDataset length: {}'.format(len(train_set)))

    shuffle(train_set)

    with CodeTimer('Training completed in {:.2f}s'):
        print('Fetching most common words...')
        common_words = most_common_words(train_set, 1000)

        print('Extracting featuresets...')
        featuresets = [(document_features(d, common_words), c) for (d, c) in tqdm(train_set)]
        del train_set

        print('Training the data (NaiveBayes)...')
        classifier = nltk.NaiveBayesClassifier.train(featuresets)
        del featuresets

    print('Testing classifiers.....')
    test_set = dataset.load(test_path)

    with CodeTimer('Classification completed in {:.2f}s'):
        test_set_features = [(document_features(d, common_words), c) for (d, c) in tqdm(test_set)]
        del test_set
        accuracy = nltk.classify.accuracy(classifier, test_set_features)

    print('Naive Bayes Accuracy: {}'.format(accuracy))

    classifier.show_most_informative_features(30)


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
