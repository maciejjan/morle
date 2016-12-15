import hfst
import logging
import sys

import algorithms.fst
import shared

def prepare_analyzer():
    analyzer = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    rootgen = algorithms.fst.load_transducer(shared.filenames['rootgen-tr'])
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])

    analyzer.minimize()
    analyzer.compose(rules_tr)
    analyzer.minimize()
    rootgen.compose(rules_tr)
    rootgen.minimize()
    analyzer.disjunct(rootgen)
    analyzer.invert()
    analyzer.convert(hfst.HFST_OLW_TYPE)

    return analyzer

def run():
    analyzer = prepare_analyzer()
    logging.getLogger('main').info('Ready.')
    for line in sys.stdin:
        word = line.rstrip()
        for base, cost in analyzer.lookup(word):
            print('\t'.join([base, str(cost)]))

