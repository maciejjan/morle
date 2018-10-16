import algorithms.fst
from datastruct.lexicon import LexiconEntry
from datastruct.rules import Rule
from utils.files import file_exists, read_tsv_file
import shared

import hfst
import logging
import math
from operator import itemgetter
from typing import List, Tuple


# TODO use RuleSet instead!
def load_rules() -> List[Tuple[Rule, float]]:
    rules_filename = None
    if shared.config['compile'].getboolean('weighted'):
        if shared.config['Models'].get('edge_model') == 'simple':
            rules_filename = shared.filenames['edge-model']
            max_cost = shared.config['compile'].getfloat('max_cost')
            rules = [(Rule.from_string(rule), -math.log(prod))\
                     for rule, prod in\
                         read_tsv_file(rules_filename, (str, float))\
                     if -math.log(prod) < max_cost ] +\
                    [(Rule.from_string(':/:___:'), 0.0)]
            return rules
        else:
            raise Exception('Compiling a weighted analyzer is only possible'
                            ' for the Bernoulli edge model.')
    else:
        rules_filename = shared.filenames['rules-modsel']
        if not file_exists(rules_filename):
            rules_filename = shared.filenames['rules']
        return [(Rule.from_string(rule), 0.0)\
                for (rule,) in read_tsv_file(rules_filename, (str,))] +\
               [(Rule.from_string(':/:___:'), 0.0)]


# TODO use Lexicon instead!
def load_roots() -> List[LexiconEntry]:

    def root_reader():
        col = 0
        for row in read_tsv_file(shared.filenames['wordlist']):
            if col < len(row) and row[col]:
                yield row[col]

    roots = []
    for root_str in root_reader():
        try:
            roots.append(LexiconEntry(root_str))
        except Exception as ex:
            logging.getLogger('main').warning(str(ex))
    return roots


def build_rule_transducer(rules :List[Tuple[Rule, float]]) \
                         -> hfst.HfstTransducer:
    transducers = []
    for rule, weight in rules:
        rule_tr = rule.to_fst(weight=weight)
        transducers.append(rule_tr)
    result = algorithms.fst.binary_disjunct(transducers, print_progress=True)
    return result


def build_root_transducer(roots :List[LexiconEntry]) -> hfst.HfstTransducer:
    transducers = []
    for root in roots:
        seq = root.word + root.tag
        transducers.append(algorithms.fst.seq_to_transducer(zip(seq, seq)))
    result = algorithms.fst.binary_disjunct(transducers, print_progress=True)
    return result


# def build_rootgen_transducer(roots :List[LexiconEntry]) -> hfst.HfstTransducer:
#     alergia = AlergiaStringFeature()
#     alergia.fit(roots)
#     return alergia.automaton


def run() -> None:
    rules = load_rules()
    roots = load_roots()

    logging.getLogger('main').info('Building the rule transducer...')
    rules_tr = build_rule_transducer(rules)
    algorithms.fst.save_transducer(rules_tr, shared.filenames['rules-tr'])

    if shared.config['General'].getboolean('supervised'):
        logging.getLogger('main').info('Building the root transducer...')
        roots_tr = build_root_transducer(roots)
        algorithms.fst.save_transducer(roots_tr, shared.filenames['roots-tr'])

#     logging.getLogger('main').info('Building the root generator transducer...')
#     rootgen_tr = algorithms.fst.load_transducer(shared.filenames['root-model'])
#     algorithms.fst.save_transducer(rootgen_tr, shared.filenames['rootgen-tr'])

