import algorithms.fst
from datastruct.lexicon import *
from datastruct.rules import *
from utils.files import *
import shared

from operator import itemgetter
import hfst
import logging
import math


def load_rules():
    return [(Rule.from_string(rule), domsize, prod)\
            for rule, domsize, prod in\
                read_tsv_file(shared.filenames['rules-fit'], (str, int, float))]


def build_rule_transducers(rules):
    max_cost = shared.config['generate'].getfloat('max_cost')
    transducers, potential_words = [], 0
    for rule, domsize, prod in rules:
        cost = -math.log(prod)
        if cost < max_cost:
            transducers.append(rule.to_fst(weight=cost))
    return transducers


def word_generator(lexicon_tr, rules_tr):
    logging.getLogger('main').info('Composing...')
    tr = hfst.HfstTransducer(lexicon_tr)
    tr.compose(rules_tr)
    tr.minimize()
    lexicon_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)

    logging.getLogger('main').info('Extracting paths...')
    for input_word, outputs in tr.extract_paths(output='dict').items():
        input_word = input_word.replace(hfst.EPSILON, '')
        input_word_unnorm = unnormalize_word(input_word)
        for output_word, weight in outputs:
            output_word = output_word.replace(hfst.EPSILON, '')
            output_word_unnorm = unnormalize_word(output_word)
            if not lexicon_tr.lookup(output_word):
                yield (output_word_unnorm, input_word_unnorm, weight)


def sort_and_deduplicate_results(results):
    results_list = sorted(list(results), key=itemgetter(2))
    known_output_words = set()
    for output_word, input_word, weight in results_list:
        if output_word not in known_output_words:
            known_output_words.add(output_word)
            yield (output_word, input_word, weight)


def run() -> None:
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    logging.getLogger('main').info('Building rule transducer...')
    rules_tr = algorithms.fst.binary_disjunct(
                   build_rule_transducers(load_rules()),
                   print_progress=True)
    logging.getLogger('main').info('Generating words...')
    with open_to_write(shared.filenames['wordgen']) as outfp:
        for output_word, input_word, weight in \
                sort_and_deduplicate_results(
                    word_generator(lexicon_tr, rules_tr)):
            write_line(outfp, (output_word, input_word, weight))

