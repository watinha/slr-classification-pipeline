import np

from keras import layers
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import tree, metrics, svm, naive_bayes, ensemble, linear_model, neural_network

slrs_files = {
    'games': {
        'argument': [ 'bibs/games/round1-todos.bib' ],
        'project_folder': 'games'
    },
    'slr': {
        'argument': [ 'bibs/slr/round1-todos.bib' ],
        'project_folder': 'slr'
    },
    'pair': {
        'argument': [ 'bibs/pair/round1-todos.bib' ],
        'project_folder': 'pair'
    },
    'illiterate': {
       'argument': [ 'bibs/illiterate/round1-others.bib' ],
       'project_folder': 'illiterate'
    },
    'mdwe':{
       'argument': [ 'bibs/mdwe/round1-acm.bib',
           'bibs/mdwe/round1-ieee.bib', 'bibs/mdwe/round1-sciencedirect.bib' ],
       'project_folder': 'mdwe'
    },
    'testing': {
       'argument': [ 'bibs/testing/round1-google.bib',
           'bibs/testing/round1-ieee.bib', 'bibs/testing/round1-outros.bib',
           'bibs/testing/round2-google.bib', 'bibs/testing/round2-ieee.bib',
           'bibs/testing/round2-outros.bib', 'bibs/testing/round3-google.bib'],
       'project_folder': 'testing'
    },
    'ontologies': {
       'argument': [ 'bibs/ontologies/round1-google.bib',
           'bibs/ontologies/round1-ieee.bib', 'bibs/ontologies/round1-outros.bib',
           'bibs/ontologies/round2-google.bib', 'bibs/ontologies/round2-ieee.bib',
           'bibs/ontologies/round3-google.bib' ],
       'project_folder': 'ontologies'
    },
    'xbi': {
       'argument': [ 'bibs/xbi/round1-google.bib',
           'bibs/xbi/round1-ieee.bib', 'bibs/xbi/round1-outros.bib',
           'bibs/xbi/round2-google.bib', 'bibs/xbi/round2-ieee.bib',
           'bibs/xbi/round3-google.bib' ],
       'project_folder': 'xbis'
   }
}

def get_slr_files(slr):
    return slrs_files[slr]['argument']

seed = 42
def get_classifier(classifier_name):
    classifier = None
    params = {}
    if (classifier_name == 'svm'):
        classifier = svm.SVC(random_state=seed, probability=True)
        params = {
            #'kernel': ['linear', 'rbf', 'poly'],
            'kernel': ['linear', 'rbf'],
            'C': [1, 10, 100],
            #'degree': [2, 3],
            'coef0': [0, 10, 100],
            'tol': [0.001, 0.1, 1],
            'class_weight': ['balanced', None]
        }
    elif (classifier_name == 'dt'):
        classifier = tree.DecisionTreeClassifier(random_state=seed)
        params = {
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }
    elif (classifier_name == 'rf'):
        classifier = ensemble.RandomForestClassifier(random_state=seed)
        params = {
            'n_estimators': [5, 10, 100],
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }
    else:
        classifier = svm.LinearSVC(random_state=seed, fit_intercept=False)
        params = {
            'C': [1, 10, 100],
            'tol': [0.001, 0.1, 1],
            'class_weight': ['balanced', None]
        }
    return classifier, params


def get_glove (word_index, embedding_dim, embedding_file):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(embedding_file) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


def get_se (word_index, embedding_dim, embedding_file):
    se_embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    not_found = []

    for word in word_index.keys():
        try:
            idx = word_index[word]
            embedding_matrix[idx] = np.array(
                se_embeddings.get_vector(word), dtype=np.float32)[:embedding_dim]
        except:
            not_found.append(word)

    print('Not in embedding: %s...' % (not_found))
    return embedding_matrix


def get_embedding_classifier (classifier_name, vocab_size, embedding_dim, maxlen, word_index, embedding_file):
    embedding_matrix = None
    if (classifier_name == 'embeddings_glove'):
        embedding_matrix = get_glove(word_index, embedding_dim, embedding_file)
    else:
        embedding_matrix = get_se(word_index, embedding_dim, embedding_file)


    def create_model (filters=32, kernel_size=3, neurons=1, trainable=True):
        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=maxlen,
                                   trainable=trainable))
        model.add(layers.Conv1D(filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(neurons, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model


    model = KerasClassifier(build_fn=create_model, epochs=250, verbose=0)
    params = {
        'filters': [32],
        'kernel_size': [2, 3],
        'neurons': [10, 20],
        'trainable': [True, False]
    }
    return model, params
