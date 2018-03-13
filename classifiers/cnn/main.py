import classifiers as cls
import tensorflow as tf
from classifiers.cnn import Transformer

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
    t = Transformer()
    X, y = t.get_corpus(cls.MOVIE_REVIEWS_TRAIN)
    pass
    #tf.app.run()