import algorithms.fst
from algorithms.negex import NegativeExampleSampler
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
import shared

import logging
import numpy as np


def run() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    logging.getLogger('main').info('Loading lexicon transducer...')
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    logging.getLogger('main').info('Loading rules...')
    rule_set = RuleSet.load(shared.filenames['rules'])

    negex_sampler = NegativeExampleSampler(lexicon, lexicon_tr, rule_set,
                                           np.array([]), np.array([]))
    negex_sampler.sample()

