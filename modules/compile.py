from datastruct.lexicon import *
from datastruct.rules import *
from models.point import *
from utils.files import *
import shared

import libhfst
import logging
from operator import itemgetter
import sys

def load_rules():
    return [(Rule.from_string(rule), -math.log(prod))\
            for rule, domsize, prod in\
                read_tsv_file(shared.filenames['rules-fit'], (str, int, float))] +\
           [(Rule.from_string(':/:___:'), 0.0)]

def build_rule_transducers(rules):
#    rules.sort(reverse=True, key=itemgetter(2))
    transducers = []
    for rule, weight in rules:
        rule.build_transducer(weight=weight)
        transducers.append(rule.transducer)
    return transducers

def run():
    logging.getLogger('main').info('Building rule transducer...')
    rules_tr = algorithms.fst.binary_disjunct(\
                   tr for tr in build_rule_transducers(load_rules()))
#    rules_tr.invert()
    algorithms.fst.save_transducer(rules_tr, shared.filenames['rules-tr'])

def cleanup():
    remove_file_if_exists(shared.filenames['rules-tr'])

