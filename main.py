import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler

from config import get_slr_files, get_classifier
from lib.bib_loader import load
from lib.text_preprocessing import FilterComposite, StopwordsFilter, LemmatizerFilter
from lib.years_split import YearsSplit


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

slr_files = get_slr_files(theme)
classifier, classifier_params = get_classifier(classifier_name)

X, y, years = load(slr_files)

pipeline = Pipeline([
    ('preprocessing', FilterComposite([
        StopwordsFilter(), LemmatizerFilter() ])),
    ('extractor', TfidfVectorizer(ngram_range=(1, int(ngram_range)))),
    ('scaler', MaxAbsScaler()),
    ('feature_selection', SelectKBest(chi2, k=int(k))),
    ('classifier', GridSearchCV(classifier, classifier_params, cv=5))
])

years_split = YearsSplit(n_split=3, years=years)
scores = cross_validate(
        pipeline, X, y, cv=years_split, groups=years,
        scoring=['f1', 'precision', 'recall', 'roc_auc'])
print(scores)
