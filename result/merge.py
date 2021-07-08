import pandas as pd

reports = [ 'fscore', 'threashold', 'missed', 'excluded' ]
themes = [ 'games', 'illiterate', 'mdwe', 'ontologies', 'pair', 'slr', 'testing', 'xbi' ]

for report in reports:
    df = None
    for theme in themes:
        filename = 'result/%s-%s.csv' % (report, theme)
        if df is None:
            df = pd.read_csv(filename)
        else:
            df = df.append(pd.read_csv(filename))

    df.to_csv('result/%s-all.csv' % (report))


