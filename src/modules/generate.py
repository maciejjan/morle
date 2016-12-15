# naive word generation:
# - sort rules according to productivity
# - for each rule:
#   - compose with lexicon
#   - extract paths
#   - for each path: if generates new word -> output it and add it to known words

from operator import itemgetter
import hfst
import logging
import math

from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
import shared

def load_rules():
    return [(Rule.from_string(rule), domsize, prod)\
            for rule, domsize, prod in\
                read_tsv_file(shared.filenames['rules-fit'], (str, int, float))]

def build_rule_transducers(rules):
    rules.sort(reverse=True, key=itemgetter(2))
    transducers, potential_words = [], 0
    for rule, domsize, prod in rules:
        if potential_words + domsize >= shared.config['generate']\
                                              .getint('max_potential_words'):
                break
        rule.build_transducer(weight=-math.log(prod))
        transducers.append(rule.transducer)
        potential_words += domsize
    return transducers

def run():
    logging.getLogger('main').info('Building lexicon transducer...')
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    lexicon.build_transducer()
    logging.getLogger('main').info('Building rule transducer...')
    rules_tr = algorithms.fst.binary_disjunct(\
                   tr for tr in build_rule_transducers(load_rules()))

    known_words = {node.key for node in lexicon.iter_nodes()}
    
    logging.getLogger('main').info('Composing...')
    tr = lexicon.transducer
    tr.compose(rules_tr)
    tr.minimize()

    with open_to_write(shared.filenames['wordgen']) as outfp:
        for input_word, outputs in tr.extract_paths(output='dict').items():
            input_word = input_word.replace(hfst.EPSILON, '')
            for output_word, weight in outputs:
                output_word = output_word.replace(hfst.EPSILON, '')
                if output_word not in known_words:
                    write_line(outfp, (output_word, input_word, weight))
                    known_words.add(output_word)

