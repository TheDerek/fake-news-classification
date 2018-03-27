import string

from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords as sw
from sklearn.base import TransformerMixin, BaseEstimator


class Tokenise(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.translator = str.maketrans('', '', string.punctuation)
        self.tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')

    def transform(self, X: [str], y=None) -> iter([str]):
        return (self._tokenise(document) for document in X)

    def fit(self, X, y=None):
        return self

    def _tokenise(self, document: str):
        document = document.lower()
        document = document.translate(self.translator)
        return self.tokenizer.tokenize(document)


class RemoveStopWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = sw.words('english')

    def transform(self, X: iter([str]), y=None) -> iter([str]):
        return (self._remove_stop_words(document) for document in X)

    def fit(self, X, y=None):
        return self

    def _remove_stop_words(self, document):
        return [word for word in document if word not in self.stopwords]


class Lemmenise(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmeniser = WordNetLemmatizer()

    def transform(self, X: iter([str]), y=None) -> iter([str]):
        return (self._lemmenise(document) for document in X)

    def fit(self, X, y=None):
        return self

    def _lemmenise(self, document: [str]) -> iter([str]):
        return [self.lemmeniser.lemmatize(word) for word in document]


def identity(x):
    return x
