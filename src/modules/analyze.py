from algorithms.align import extract_all_rules
import algorithms.fst
from datastruct.lexicon import Lexicon
from utils.files import file_exists, open_to_write, read_tsv_file, write_line
from utils.printer import progress_printer
import shared

import hfst
import logging


def prepare_analyzer(lexicon):
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    logging.getLogger('main').info('Building lexicon transducer...')
    lexicon.build_transducer(print_progress=True)
    analyzer = hfst.HfstTransducer(lexicon.transducer)
    logging.getLogger('main').info('Composing with rules...')
    analyzer.compose(rules_tr)
    analyzer.minimize()
    logging.getLogger('main').info('Composing again with lexicon...')
    analyzer.compose(lexicon.transducer)
    analyzer.minimize()
    analyzer.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    return analyzer

def load_rules():
    filename = shared.filenames['rules-modsel']
    if not file_exists(filename):
        filename = shared.filename['rules']
    rules = set()
    for (rule,) in read_tsv_file(filename, types=(str,)):
        rules.add(rule)
    return rules

def analyze(lexicon, rules, analyzer):
    logging.getLogger('main').info('Analyzing...')
    pp = progress_printer(len(lexicon))
    with open_to_write(shared.filenames['analyze.graph']) as outfp:
        for node in lexicon.iter_nodes():
            similar_words = set([word for word, cost in\
                                          analyzer.lookup(node.key)\
                                          if node.key < word])
            for word in similar_words:
                for rule in extract_all_rules(node, lexicon[word]):
                    if str(rule) in rules:
                        write_line(outfp, (node.key, word, str(rule)))
                    rule = rule.reverse()
                    if str(rule) in rules:
                        write_line(outfp, (word, node.key, str(rule)))
            next(pp)

def run():
    lexicon = Lexicon.init_from_wordlist(shared.filenames['analyze.wordlist'])
    analyzer = prepare_analyzer(lexicon)
    rules = load_rules()
    analyze(lexicon, rules, analyzer)

