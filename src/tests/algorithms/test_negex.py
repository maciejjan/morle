from algorithms.negex import NegativeExampleSampler
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import Rule, RuleSet
from datastruct.graph import GraphEdge, EdgeSet
import shared

import unittest


# fake config file
CONFIG = '''
[General]
encoding = utf-8
date_format = %%d.%%m.%%Y %%H:%%M
supervised = no

[Models]
root_feature_model = none
edge_feature_model = none

[NegativeExampleSampler]
num_processes = 1
'''

shared.config.read_string(CONFIG)


class NegativeExampleSamplerTest(unittest.TestCase):

    def test_complete_sample(self) -> None:
        'Test a sample consisting of all possible negative edges.'
        lexicon = Lexicon()
        lexicon.add(LexiconEntry('machen'))
        lexicon.add(LexiconEntry('Sachen'))
        rule_set = RuleSet()
        rule_set.add(Rule.from_string(':/en:t___:'), 2)
        # TODO compute domsizes!!!
        edge_set = EdgeSet()
        negex_sampler = NegativeExampleSampler(lexicon, rule_set, edge_set)
        # TEST DATA:
        # Lexicon:
        # RuleSet:
        # positive edges:
        # expected negative edges:

    def test_random_sample(self) -> None:
        'Test a random sample in case there are many possible negative edges.'
        pass
        # TEST DATA:
        # Lexicon:
        # RuleSet:
        # positive edges:
        # test for a valid negative edge:
        # - source in lexicon
        # - target not in lexicon
        # - rule in RuleSet
        # - "target" in transducer source .o. rule
        # - weights: [TODO values]

