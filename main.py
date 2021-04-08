import sys, np, pandas as pd

#from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing.text import Tokenizer

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler

from config import get_slr_files, get_classifier, get_extractor#, get_embedding_classifier
from lib.bib_loader import load
from lib.text_preprocessing import FilterComposite, StopwordsFilter, LemmatizerFilter
from lib.years_split import YearsSplit


if (len(sys.argv) < 2):
    print('second argument missing: SLR theme (games|slr|illiterate|pair|mdwe|testing|ontologies|xbi)')
    sys.exit(1)

if (len(sys.argv) < 3):
    print('third argument missing: classifier (svm|dt|rf|lsvm|embeddings_glove|embeddings_se)')
    sys.exit(1)

if (len(sys.argv) < 4):
    print('forth argument missing: ngram range (1:5)')
    sys.exit(1)

if (len(sys.argv) < 5):
    print('fifth argument missing: titles only?')
    sys.exit(1)

if (len(sys.argv) < 6):
    print('sixth argument missing: padding sequence (for embeddings only!)')
    sys.exit(1)

if (len(sys.argv) < 7):
    print('seventh argument missing: extrator (tfidf,embeddings_glove,embeddings_se)')
    sys.exit(1)

_, theme, classifier_name, ngram_range, titles, maxlen, extractor = sys.argv
titles = True if titles == 'true' else False
embedding_dim = 200
embedding_file = './embeddings/glove.6B.200d.txt' if classifier_name == 'embeddings_glove' or extractor == 'embeddings_glove' else './embeddings/SO_vectors_200.bin'

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

    if classifier_name[:9] != 'embedding':
        classifier, classifier_params = get_classifier(classifier_name)
        extractor_class, selector_f, selector_params = get_extractor(extractor, ngram_range, embedding_file)
        classifier_params.update(selector_params)
        classifier_pipeline = GridSearchCV(Pipeline([
            ('feature_selection', SelectKBest(selector_f)),
            ('classifier', classifier)
        ]), classifier_params, cv=5, scoring='accuracy')
        pipeline = Pipeline([
            ('preprocessing', FilterComposite([
                StopwordsFilter(), LemmatizerFilter() ])),
            ('extractor', extractor_class),
            ('scaler', MaxAbsScaler()),
            ('classifier', classifier_pipeline)
        ])

    else: # classifier_name == 'embeddings_glove' or 'embeddings_se'
        pass
        #tokenizer = Tokenizer(num_words=int(k))
        #tokenizer.fit_on_texts(X_train)
        #X_train = tokenizer.texts_to_sequences(X_train)
        #X_train = pad_sequences(X_train, padding='post', maxlen=int(maxlen))
        #X_test = tokenizer.texts_to_sequences(X_test)
        #X_test = pad_sequences(X_test, padding='post', maxlen=int(maxlen))
        #classifier, classifier_params = get_embedding_classifier(classifier_name, len(tokenizer.word_index) + 1,
        #        embedding_dim, int(maxlen), tokenizer.word_index, embedding_file)
        #pipeline = GridSearchCV(classifier, classifier_params, cv=5, scoring='accuracy')

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
    column_name = '%s-%s' % (classifier_name, extractor)
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
