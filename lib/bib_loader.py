import codecs, bibtexparser

class BibLoader():
    def __init__ (self):
        pass

    def load(self, file_list):
        X = []
        y = []
        years = []
        for f in file_list:
            with codecs.open(f, 'r', encoding='utf-8') as bib_f:
                db = bibtexparser.load(bib_f)
                for bib_index, entry in enumerate(db.entries, start=0):
                    label = 1 if entry['inserir'] == 'true' else 0
                    title = entry['title']
                    abstract = entry['abstract']
                    year = int(entry['year'])
                    content = u'%s\n%s' % (title, abstract)
                    X.append(content)
                    y.append(label)
                    years.append(year)
        return X, y, years
