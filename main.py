import sys, np

from sklearn import metrics
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
X, y, years = load(slr_files)


kfold = YearsSplit(n_split=3, years=years)
correct_exclusion_rate = []
threasholds = []
missed = []
fscore_threashold = []

X = np.array(X)
y = np.array(y)
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier, classifier_params = get_classifier(classifier_name)
    pipeline = Pipeline([
        ('preprocessing', FilterComposite([
            StopwordsFilter(), LemmatizerFilter() ])),
        ('extractor', TfidfVectorizer(ngram_range=(1, int(ngram_range)))),
        ('scaler', MaxAbsScaler()),
        ('feature_selection', SelectKBest(chi2, k=int(k))),
        ('classifier', GridSearchCV(classifier, classifier_params, cv=5))
    ])

    pipeline.fit(X_train, y_train)
    y_score = pipeline.predict_proba(X_train)[:, 1]
    precision, recall, threasholds2 = metrics.precision_recall_curve(
            y_train, y_score)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    matrix = metrics.confusion_matrix(
            y_test, [ 0 if i < threasholds2[0] else 1 for i in y_score ])
    correct_exclusion_rate.append(
            matrix[0, 0] /
            (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
    missed.append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))
    threasholds.append(threasholds2[0])
    fscore_threashold.append(metrics.f1_score(
        y_test, [ 0 if i < threasholds2[0] else 1 for i in y_score ]))

print(fscore_threashold)
print(threasholds)
print(missed)
print(correct_exclusion_rate)
