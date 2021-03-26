import sys

from lib.bib_loader import BibLoader
from sklearn import tree, metrics, svm, naive_bayes, ensemble, linear_model, neural_network

slrs = {
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

if (len(sys.argv) < 2):
    print('second argument missing: SLR theme (games|slr|illiterate|pair|mdwe|testing|ontologies|xbi)')
    sys.exit(1)

if (len(sys.argv) < 3):
    print('third argument missing: classifier (svm|dt|nn|rf|nb|lr|lsvm)')
    sys.exit(1)

if (len(sys.argv) < 4):
    print('forth argument missing: number of features')
    sys.exit(1)

theme, classifier, k = sys.argv[1:]
seed = 42

params = {}
if (classifier == 'svm'):
    classifier = svm.SVC(random_state=seed, probability=True)
    params = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [1, 10, 100],
        'degree': [1, 2, 3],
        'coef0': [0, 10, 100],
        'tol': [0.001, 0.1, 1],
        'class_weight': ['balanced', None]
    }
elif (classifier == 'dt'):
    classifier = tree.DecisionTreeClassifier(random_state=seed)
    params = {
        'criterion': ["gini", "entropy"],
        'max_depth': [10, 50, 100, None],
        'min_samples_split': [2, 10, 100],
        'class_weight': [None, 'balanced']
    }
elif (classifier == 'rf'):
    classifier = ensemble.RandomForestClassifier(random_state=seed)
    params = {
        'n_estimators': [5, 10, 100],
        'criterion': ["gini", "entropy"],
        'max_depth': [10, 50, 100, None],
        'min_samples_split': [2, 10, 100],
        'class_weight': [None, 'balanced']
    }
else:
    classifier = svm.LinearSVC(random_state=seed)
    params = {
        'C': [1, 10, 100],
        'tol': [0.001, 0.1, 1],
        'class_weight': ['balanced', None]
    }


X, y, years = (BibLoader()).load(slrs[theme]['argument'])

