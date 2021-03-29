import sys

from sklearn.feature_extraction.text import TfidfVectorizer

from config import get_slr_files, get_classifier
from lib.bib_loader import load


if (len(sys.argv) < 2):
    print('second argument missing: SLR theme (games|slr|illiterate|pair|mdwe|testing|ontologies|xbi)')
    sys.exit(1)

if (len(sys.argv) < 3):
    print('third argument missing: classifier (svm|dt|nn|rf|nb|lr|lsvm)')
    sys.exit(1)

if (len(sys.argv) < 4):
    print('forth argument missing: number of features')
    sys.exit(1)

if (len(sys.argv) < 5):
    print('forth argument missing: ngram range (1:5)')
    sys.exit(1)

_, theme, classifier_name, k, ngram_range = sys.argv
seed = 42

slr_files = get_slr_files(theme)
classifier = get_classifier(classifier_name)

X, y, years = load(slr_files)
