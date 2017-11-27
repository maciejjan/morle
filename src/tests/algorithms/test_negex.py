import algorithms.negex
import shared

import unittest


# fake config file
CONFIG = '''
[NegativeExampleSampler]
num_processes = 1
'''

shared.config.read_string(CONFIG)


class NegativeExampleSamplerTest(unittest.TestCase):

    def test_complete_sample(self) -> None:
        'Test a sample consisting of all possible negative edges.'
        pass
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

