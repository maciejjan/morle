# naive analysis:
# - lookup the word in the ruleset transducer
# - lemma cost = 0 if lemma in lexicon else rootdist(lemma)
# - also: use training data as stored lemmas

from datastruct.lexicon import *
from datastruct.rules import *
from models.point import *
import shared

import libhfst
import logging
from operator import itemgetter
import sys

def load_rules():
    return [(Rule.from_string(rule), domsize, prod)\
            for rule, domsize, prod in\
                read_tsv_file(shared.filenames['rules-fit'], (str, int, float))]

def build_rule_transducers(rules):
#    rules.sort(reverse=True, key=itemgetter(2))
    transducers = []
    for rule, domsize, prod in rules:
        rule.build_transducer(weight=-math.log(prod))
        transducers.append(rule.transducer)
    return transducers

def analyze(word, lexicon, rules_tr, model):
    def root_cost(node):
        return model.rootdist.cost_of_change(\
                model.extractor.extract_feature_values_from_nodes((node,)), ())

    tr = algorithms.fst.seq_to_transducer(word.seq())
    tr.compose(rules_tr)
    results = [('___',
                0 if word.key in lexicon else root_cost(word))]
    for base, cost in sum(tr.extract_paths(output='dict').values(), []):
        base = base.replace(libhfst.EPSILON, '')
        try:
            results.append((lexicon[base].key, cost))
        except KeyError:
            n_base = LexiconNode(base)
            results.append(('?'+n_base.key, cost + root_cost(n_base)))
    results.sort(key=itemgetter(1))
    return results[:shared.config['analyze'].getint('max_results')]

def run():
    logging.getLogger('main').info('Building rule transducer...')
    rules_tr = algorithms.fst.binary_disjunct(\
                   tr for tr in build_rule_transducers(load_rules()))
    rules_tr.invert()
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    model = PointModel(lexicon)

    logging.getLogger('main').info('Ready.')
    for line in sys.stdin:
        word = LexiconNode(line.rstrip())
#        print('analyzing...')
        analysis = analyze(word, lexicon, rules_tr, model)
        for base, cost in analysis:
#            print('\t'.join(word.key(), base.key(), cost))
#            base = base.replace(libhfst.EPSILON, '')
            print('\t'.join((word.key, base, str(cost))))
#        print('finished')
