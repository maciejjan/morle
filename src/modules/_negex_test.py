import algorithms.fst
from algorithms.negex import NegativeExampleSampler
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from utils.files import open_to_write, write_line
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
    with open_to_write('negex.txt') as fp:
        for edge in negex_sampler.sample(600000):
            write_line(fp, (edge.source, edge.target, edge.rule))

