import newspaper
import scipy.misc

from matplotlib import pyplot as plt
from classifiers.cnn import Dataset
from scipy import ndimage

url = 'https://www.bloomberg.com/view/articles/2018-03-15/is-insider-guessing-illegal'

if __name__ == "__main__":
    article = newspaper.Article(url)
    article.download()
    article.parse()
    text = article.text

    d = Dataset()
    embedding = d.get_embeddings(text)[:300, :]
    blurred = ndimage.gaussian_filter(embedding, sigma=7)

    scipy.misc.imsave('raw.png', embedding)
    scipy.misc.imsave('blurred.png', blurred)
