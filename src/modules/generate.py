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
    max_cost = shared.config['generate'].getfloat('max_cost')
    rules.sort(reverse=True, key=itemgetter(2))
    transducers, potential_words = [], 0
    for rule, domsize, prod in rules:
        cost = -math.log(prod)
        if cost < max_cost:
            transducers.append(rule.to_fst(weight=-math.log(prod)))
    return transducers

def word_generator(lexicon):
    logging.getLogger('main').info('Building rule transducer...')
    rules_tr = algorithms.fst.binary_disjunct(\
                   tr for tr in build_rule_transducers(load_rules()))

    known_words = {entry.symstr for entry in lexicon.entries()}
    
    logging.getLogger('main').info('Composing...')
    tr = lexicon.to_fst()
    tr.compose(rules_tr)
    tr.minimize()

    for input_word, outputs in tr.extract_paths(output='dict').items():
        input_word = input_word.replace(hfst.EPSILON, '')
        for output_word, weight in outputs:
            output_word = output_word.replace(hfst.EPSILON, '')
            if output_word not in known_words:
                yield (unnormalize_word(output_word), 
                       unnormalize_word(input_word), 
                       weight)


def run_without_eval() -> None:
    wordgen = word_generator()
    with open_to_write(shared.filenames['wordgen']) as outfp:
        for output_word, input_word, weight in wordgen:
            write_line(outfp, (output_word, input_word, weight))


def run_with_eval() -> None:
    logging.getLogger('main').info('Building lexicon transducer...')
    lexicon = Lexicon(filename=shared.filenames['wordlist'])
    known_words = {entry.symstr for entry in lexicon.entries()}
    # output: word base cost precision recall f-score
    # (compute running precision, recall and f-score)
    wordgen = word_generator(lexicon)
    generated_words = [(input_word, output_word, cost)\
                       for (input_word, output_word, cost) in wordgen]
    generated_words.sort(key=itemgetter(2))
    eval_words = set(word for (word,) in read_tsv_file(shared.filenames['eval.wordlist'])) -\
                 known_words
    tp, fp, fn = 0, 0, len(eval_words)
    with open_to_write(shared.filenames['eval.wordgen']) as outfp:
        for output_word, input_word, weight in generated_words:
            current_result = None
            if output_word in eval_words:
                tp += 1
                fn -= 1
                current_result = '+'
            else:
                fp += 1
                current_result = '-'
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            fsc = 2 / (1 / precision + 1 / recall) \
                  if precision*recall != 0.0 else 0.0
            results = '{}\t{:.1%}\t{:.1%}\t{:.1%}'.format(
                        current_result, precision, recall, fsc)
            write_line(outfp, (output_word, input_word, weight, results))


def run() -> None:
    if shared.config['generate'].getboolean('evaluate'):
        run_with_eval()
    else:
        run_without_eval()

