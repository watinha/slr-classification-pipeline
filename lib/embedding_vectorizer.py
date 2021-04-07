import np

from nltk.tokenize import word_tokenize

class AverageEmbeddingVectorizer():

    def __init__(self, loader):
        self._loader = loader

    def fit (self, X, y):
        return self

    def fit_transform (self, X, y):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        sentences = [word_tokenize(row) for row in X]
        word_list = set(sum(sentences, []))
        word_index = self._loader.build_word_index(word_list)

        result = []
        for sentence in sentences:
            vec = None
            n_words = 0
            for word in sentence:
                if word in word_index:
                    if vec is None:
                        vec = word_index[word]
                    else:
                        vec = vec + word_index[word]
                    n_words = n_words + 1
            result.append(vec / n_words)

        return np.array(result)


class GloveLoader:

    def __init__ (self, filename):
        self._filename = filename

    def build_word_index (self, word_list):
        word_index = {}
        with open(self._filename) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_list:
                    word_index[word] = np.array(
                            vector, dtype=np.float32)
        return word_index


