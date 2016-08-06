import algorithms.ngrams
from datastruct.lexicon import *
from datastruct.rules import *
from models.features.marginal import *
from models.marginal import *
import settings

import unittest

class MarginalModelFeaturesTest(unittest.TestCase):
    words = ['this', 'is', 'a', 'simple', 'test', 'suite', 'consisting',\
        'of', 'a', 'couple', 'of', 'tests']
    rules = [':/:s']

    def test_string_feature(self):
        '''Create a MarginalStringFeature, add some values to it and check
           the statistics.'''
        feature = MarginalStringFeature()
        feature.fit(algorithms.ngrams.generate_n_grams(\
            tuple(word) + ('#',), 1) for word in self.words)
        feature.apply_change(list(algorithms.ngrams.generate_n_grams(\
            tuple(word) + ('#',), 1) for word in self.words), 
            [])
        self.assertEqual(feature.counts[('i',)], 6)
        self.assertAlmostEqual(feature.cost(), 148.398, places=3)

    def test_gaussian_inverse_chi_squared_feature(self):
        feature = MarginalGaussianInverseChiSquaredFeature(1, 1, 0, 1, 1)
        values_to_add = [[1,2,3], [1,3,2]]
        values_to_remove = []
        cost_1 = feature.cost_of_change(values_to_add, values_to_remove)
        feature.apply_change(values_to_add, values_to_remove)
        values_to_remove = values_to_add
        values_to_add = []
        cost_2 = feature.cost_of_change(values_to_add, values_to_remove)
        feature.apply_change(values_to_add, values_to_remove)

        self.assertAlmostEqual(cost_1, -cost_2)
        self.assertEqual(feature.cost(), 0)
    
    def test_marginal_model(self):

        def build_lexicon(words):
            lexicon = Lexicon()
            for word in words:
                lexicon.add_node(LexiconNode(word))
            return lexicon

        lexicon = build_lexicon(self.words)
        ruleset = set(map(Rule.from_string, self.rules))
        model = MarginalModel(lexicon, ruleset)

        edge = LexiconEdge(lexicon['test'], lexicon['tests'], list(ruleset)[0])
        cost_1 = model.cost_of_change([edge], [])
        model.apply_change([edge], [])
        cost_2 = model.cost_of_change([], [edge])
        self.assertAlmostEqual(cost_1, -cost_2)

