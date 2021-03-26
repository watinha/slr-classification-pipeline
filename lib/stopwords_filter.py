from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class StopWordsFilter ():

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_token = [ word_tokenize(row.lower()) for row in X ]
        X_filtered = []
        for tokens in X_token:
            X_filtered.append([
                token for token in tokens
                      if token not in stopwords.words('english')])
        return [ ' '.join(row) for row in  X_filtered]



    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
