from unittest import TestCase
from unittest.mock import Mock

from lib.stopwords_filter import StopWordsFilter

class StopWordsFilterTest (TestCase):

    def test_filter_implements_pipeline (self):
        X = []
        fil = StopWordsFilter()
        s = fil.fit(X)
        res = fil.transform(X)

        self.assertEqual(fil, s)
        self.assertEqual([], res)

        s = fil.fit(X, [])
        res = fil.transform(X, [])

        self.assertEqual(fil, s)
        self.assertEqual([], res)

    def test_fit_transform(self):
        X = []
        fil = StopWordsFilter()
        fil.fit = Mock(return_value=fil)
        fil.transform = Mock(return_value=[1, 2, 3])

        res = fil.fit_transform(X)

        fil.fit.assert_called_once_with(X)
        fil.transform.assert_called_once_with(X)

        self.assertEqual([1, 2, 3], res)

        fil.fit = Mock(return_value=fil)
        fil.transform = Mock(return_value=[1, 2, 3])

        res = fil.fit_transform(X, [])

        fil.fit.assert_called_once_with(X)
        fil.transform.assert_called_once_with(X)

        self.assertEqual([1, 2, 3], res)


    def test_transform_removes_stopwords (self):
        X = ['Some other area of testing a single approach with twinkles',
             'Another Super Example of Fish under the Table with ketchup']
        fil = StopWordsFilter()

        res = fil.transform(X)

        self.assertEqual([
            'area testing single approach twinkles',
            'another super example fish table ketchup'], res)
