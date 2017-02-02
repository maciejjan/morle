# naive word generation:
# - sort rules according to productivity
# - for each rule:
#   - compose with lexicon
#   - extract paths
#   - for each path: if generates new word -> output it and add it to known words

# TODO use precompiled transducers
# TODO include frequency class in the generation
# TODO generate also word features

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
#         if potential_words + domsize >= shared.config['generate']\
#                                               .getint('max_words'):
#                 break
        cost = -math.log(prod)
        if cost < max_cost:
            rule.build_transducer(weight=-math.log(prod))
            transducers.append(rule.transducer)
#         potential_words += domsize
    return transducers

def word_generator():
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

#     count, max_count = 0, shared.config['generate'].getint('max_words')
    for input_word, outputs in tr.extract_paths(output='dict').items():
        input_word = input_word.replace(hfst.EPSILON, '')
        for output_word, weight in outputs:
            output_word = output_word.replace(hfst.EPSILON, '')
            if output_word not in known_words:
                yield (output_word, input_word, weight)
#                 count += 1
#                 if count >= max_count:
#                     return
#                 known_words.add(output_word)

def run():
    wordgen = word_generator()
    with open_to_write(shared.filenames['wordgen']) as outfp:
        for output_word, input_word, weight in wordgen:
            write_line(outfp, (output_word, input_word, weight))

def eval():
    # output: word base cost precision recall f-score
    # (compute running precision, recall and f-score)
    wordgen = word_generator()
    tp, fp, fn = 0, 0, len(eval_words)
    with open_to_write(shared.filenames['eval.wordlist']) as outfp:
        for output_word, input_word, weight in wordgen:
            if output_word in eval_words:
                tp += 1
                fn -= 1
            else:
                fp += 1
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            fsc = 2 / (1 / precision + 1 / recall)
            results = '{:.1}\t{:.1}\t{:.1}'.format(
                        precision*100, recall*100, fsc*100)
            write_line(outfp, (output_word, input_word, weight, results))

