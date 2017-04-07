import algorithms.fst
from datastruct.lexicon import LexiconEntry
from datastruct.rules import Rule
from models.features.generic import AlergiaStringFeature
from utils.files import read_tsv_file
import shared

import hfst
import logging
import math
from operator import itemgetter
from typing import List, Tuple

# TODO compile the following automata:
# - rules
# - rootgen
# if supervised or bipartite:
# - roots acceptor

def load_rules() -> List[Tuple[Rule, float]]:
    return [(Rule.from_string(rule), -math.log(prod))\
            for rule, domsize, prod in\
                read_tsv_file(shared.filenames['rules-fit'],
                              (str, int, float))] +\
           [(Rule.from_string(':/:___:'), 0.0)]


def load_roots() -> List[LexiconEntry]:

    def root_reader():
        col = 0
        for row in read_tsv_file(shared.filenames['wordlist']):
            if col < len(row) and row[col]:
                yield row[col]

    roots = []
    for root_str in root_reader():
        try:
            roots.append(root_str)
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
# TODO only if supervised
    transducers = []
    for root in roots:
        transducers.append(algorithms.fst.seq_to_transducer(root.seq()))
    result = algorithms.fst.binary_disjunct(transducers, print_progress=True)
    return result


def build_rootgen_transducer(roots :List[LexiconEntry]) -> hfst.HfstTransducer:
    alergia = AlergiaStringFeature()
    alergia.fit(roots)
    return alergia.automaton


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

    logging.getLogger('main').info('Building the root generator transducer...')
    rootgen_tr = build_rootgen_transducer(roots)
    algorithms.fst.save_transducer(rootgen_tr, shared.filenames['rootgen-tr'])

