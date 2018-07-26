from algorithms.analyzer import Analyzer
import algorithms.fst
from datastruct.lexicon import *
from datastruct.rules import *
from models.suite import ModelSuite
from utils.files import *
import shared

from operator import itemgetter
import hfst
import logging
import math


# TODO
# - generate from all rules, regardless of the cost
# - additionally: word frequency test
#   - fit the frequency model using a sampler
#   - (result: a mean and sdev for each rule)
#   - add the cost of a frequency below 1 to the cost of the word
# - sum the probabilities of a word over all possible derivations!
#   (pairs: source_word, rule)
# - result: pairs (word, cost)


# def load_rules():
#     return [(Rule.from_string(rule), domsize, prod)\
#             for rule, domsize, prod in\
#                 read_tsv_file(shared.filenames['rules-fit'], (str, int, float))]
# 
# 
# def build_rule_transducers(rules):
#     max_cost = shared.config['generate'].getfloat('max_cost')
#     transducers = []
#     for rule, domsize, prod in rules:
#         cost = -math.log(prod)
#         if cost < max_cost:
#             transducers.append(rule.to_fst(weight=cost))
#     return transducers
# 
# 
# def word_generator(lexicon_tr, rules_tr):
#     lexicon_words = set(input_word \
#                         for input_word, outputs in \
#                             lexicon_tr.extract_paths(output='dict').items())
#     logging.getLogger('main').info('Composing...')
#     tr = hfst.HfstTransducer(lexicon_tr)
#     tr.compose(rules_tr)
#     tr.minimize()
# #     lexicon_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
#     tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
# 
#     logging.getLogger('main').info('Extracting paths...')
# #     for input_word, outputs in tr.extract_paths(output='dict').items():
#     for input_word in lexicon_words:
#         outputs = tr.lookup(input_word)
#         input_word = input_word.replace(hfst.EPSILON, '')
#         input_word_unnorm = unnormalize_word(input_word)
#         for output_word, weight in outputs:
#             output_word = output_word.replace(hfst.EPSILON, '')
#             output_word_unnorm = unnormalize_word(output_word)
#             if output_word_unnorm not in lexicon_words:
#                 yield (output_word_unnorm, input_word_unnorm, weight)
# 
# 
# def sort_and_deduplicate_results(results):
#     results_list = sorted(list(results), key=itemgetter(2))
#     known_output_words = set()
#     for output_word, input_word, weight in results_list:
#         if output_word not in known_output_words:
#             known_output_words.add(output_word)
#             yield (output_word, input_word, weight)


def run() -> None:
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    analyzer = Analyzer(lexicon, model)
    new_words_acceptor = hfst.HfstTransducer(analyzer.fst)
    new_words_acceptor.convert(hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    new_words_acceptor.input_project()
    new_words_acceptor.minimize()
    new_words_acceptor.subtract(lexicon.to_fst())
    new_words_acceptor.minimize()
    tr_path = full_path('wordgen.fst')
    algorithms.fst.save_transducer(new_words_acceptor, tr_path)

#     means, sdevs = fit_frequency_model(full_graph, model)

    cmd = ['hfst-fst2strings', tr_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, 
                         universal_newlines=True, bufsize=1)
    with open_to_write(shared.filenames['wordgen']) as fp:
        while True:
            line = p.stdout.readline().strip()
            if line:
                word = line.rstrip()
                analyses = analyzer.analyze(LexiconEntry(word))
                if not analyses:
                    continue
                total_cost = \
                    -math.log(sum(math.exp(-e.attr['cost']) for e in analyses))
                if total_cost < shared.config['generate'].getfloat('max_cost'):
                    write_line(fp, (word, total_cost))
            else:
                break

