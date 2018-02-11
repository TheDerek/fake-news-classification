from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

import pandas as pd

tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
stemmer = SnowballStemmer("english")
stop_words = stopwords.words('english')
stop_words += ['tass', 'jazeera']


def get_words(text, stop_words):
    text = tokenizer.tokenize(text)
    return [stemmer.stem(w.lower()) for w in text if w.lower() not in stop_words]


def apply_transform(document):
    document['text_words'] = get_words(document['text'], stop_words)
    return document, document.pop('label', None)


def load(path):
    train_set = pd.read_csv(path).T.to_dict().values()
    train_set = [apply_transform(d) for d in train_set]
    return list(train_set)
