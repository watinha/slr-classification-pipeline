import sys, np, pandas as pd

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

if (len(sys.argv) < 6):
    print('fifth argument missing: titles only?')
    sys.exit(1)

_, theme, classifier_name, k, ngram_range, titles = sys.argv
titles = tiles == 'true' ? True : False

slr_files = get_slr_files(theme)
X, y, years = load(slr_files, titles_only=titles)


kfold = YearsSplit(n_split=3, years=years)
result = {
    'fscore': [],
    'threashold': [],
    'missed': [],
    'excluded': []
}

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
        ('classifier', GridSearchCV(classifier, classifier_params, cv=5, scoring='accuracy'))
    ])

    pipeline.fit(X_train, y_train)
    y_score = pipeline.predict_proba(X_train)[:, 1]
    precision, recall, threasholds = metrics.precision_recall_curve(
            y_train, y_score)

    threashold = min(threasholds[0], 0.5)

    y_score = pipeline.predict_proba(X_test)[:, 1]
    matrix = metrics.confusion_matrix(
            y_test, [ 0 if i < threashold else 1 for i in y_score ])
    result['fscore'].append(metrics.f1_score(
        y_test, [ 0 if i < threashold else 1 for i in y_score ]))
    result['threashold'].append(threashold)
    result['excluded'].append(
            matrix[0, 0] /
            (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
    result['missed'].append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))


reports = [ 'fscore', 'threashold', 'missed', 'excluded' ]
for report in reports:
    column_name = '%s-k%d' % (classifier_name, int(k))
    labels = [('%s-%d' % (theme, i)) for i in range(3)]
    report_filename = 'result/%s-%s.csv' % (report, theme)
    df = pd.DataFrame({ column_name: result[report] }, index=labels)
    try:
        prior_df = pd.read_csv(report_filename, index_col=0)
        new_df = prior_df.join(df)
        new_df.to_csv(report_filename)
    except FileNotFoundError:
        df.to_csv(report_filename)

sys.exit(0)
