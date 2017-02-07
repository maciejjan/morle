import algorithms.alergia
import algorithms.fst
from datastruct.rules import Rule
from datastruct.lexicon import LexiconNode
from utils.files import read_tsv_file
import shared

import hfst
import logging
import math
from operator import itemgetter

def load_rules():
    return [(Rule.from_string(rule), -math.log(prod))\
            for rule, domsize, prod in\
                read_tsv_file(shared.filenames['rules-fit'],
                              (str, int, float))] +\
           [(Rule.from_string(':/:___:'), 0.0)]

def load_roots():

    def root_reader():
        col = 0 # if shared.config['General'].getboolean('supervised') else 1
        for row in read_tsv_file(shared.filenames['wordlist']):
            if col < len(row) and row[col]:
                yield row[col]

    roots = []
    for root_str in root_reader():
        try:
            roots.append(LexiconNode(root_str))
        except Exception as ex:
            logging.getLogger('main').warning(str(ex))
    return roots

def build_rule_transducer(rules):
    transducers = []
    for rule, weight in rules:
        rule.build_transducer(weight=weight)
        transducers.append(rule.transducer)
    result = algorithms.fst.binary_disjunct(transducers, print_progress=True)
    return result

def build_root_transducer(roots):
    transducers = []
    for root in roots:
        transducers.append(algorithms.fst.seq_to_transducer(root.seq()))
    result = algorithms.fst.binary_disjunct(transducers, print_progress=True)
    return result

def build_rootgen_transducer(roots):
    word_seqs = [(root.word, 1) for root in roots]
    tag_seqs = [(root.tag, 1) for root in roots]

    word_pta = algorithms.alergia.prefix_tree_acceptor(word_seqs)
    alpha = shared.config['compile'].getfloat('alergia_alpha')
    freq_threshold = shared.config['compile'].getint('alergia_freq_threshold')
    automaton = algorithms.alergia.alergia(word_pta, alpha=alpha, 
                                           freq_threshold=freq_threshold)

    tag_automaton = hfst.HfstTransducer(
                      algorithms.alergia.normalize_weights(
                        algorithms.alergia.prefix_tree_acceptor(tag_seqs)))
    tag_automaton.minimize()

    result = hfst.HfstTransducer(automaton)
    result.concatenate(tag_automaton)
    result.minimize()
    return result

def run():
    logging.getLogger('main').info('Building the rule transducer...')
    rules = load_rules()
    rules_tr = build_rule_transducer(rules)

    roots = load_roots()
    logging.getLogger('main').info('Building the root transducer...')
    roots_tr = build_root_transducer(roots)
    logging.getLogger('main').info('Building the root generator transducer...')
    rootgen_tr = build_rootgen_transducer(roots)

    logging.getLogger('main').info('Saving...')
    algorithms.fst.save_transducer(rules_tr, shared.filenames['rules-tr'])
    algorithms.fst.save_transducer(roots_tr, shared.filenames['roots-tr'])
    algorithms.fst.save_transducer(rootgen_tr, shared.filenames['rootgen-tr'])

def cleanup():
    remove_file_if_exists(shared.filenames['rules-tr'])
    remove_file_if_exists(shared.filenames['roots-tr'])

