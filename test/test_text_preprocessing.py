from unittest import TestCase
from unittest.mock import Mock

from lib.text_preprocessing import FilterStrategy, StopwordsFilter

class FilterStrategyTest (TestCase):

    def test_filter_implements_pipeline (self):
        X = []
        fil = FilterStrategy(StopwordsFilter())
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
        fil = FilterStrategy(StopwordsFilter())
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
        fil = FilterStrategy(StopwordsFilter())

        res = fil.transform(X)

        self.assertEqual([
            'area testing single approach twinkles',
            'another super example fish table ketchup'], res)
