# from algorithms.align import extract_all_rules
from algorithms.analyzer import Analyzer
import algorithms.fst
from datastruct.lexicon import Lexicon
from models.neural import ModelSuite
# from datastruct.rules import Rule
from utils.files import file_exists, open_to_write, read_tsv_file, write_line
import shared

import hfst
import logging
import tqdm
# from typing import Set

# TODO generic workflow:
# - get all analyses for a given word (with lookup)
# - compute score for each analysis

# special cases:
# - no feature model -> quicker analysis with compose + minimize + lookup
# - tag prediction


def prepare_analyzer(lexicon :Lexicon) -> hfst.HfstTransducer:
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    logging.getLogger('main').info('Building lexicon transducer...')
    lexicon_tr = lexicon.to_fst()
    analyzer = hfst.HfstTransducer(lexicon_tr)
    logging.getLogger('main').info('Composing with rules...')
    analyzer.compose(rules_tr)
    analyzer.minimize()
#     logging.getLogger('main').info('Composing again with lexicon...')
#     analyzer.compose(lexicon_tr)
#     analyzer.minimize()
    analyzer.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    return analyzer


# def load_rule_set() -> Set[Rule]:
#     filename = shared.filenames['rules-fit']
#     if not file_exists(filename):
#         filename = shared.filenames['rules-modsel']
#     if not file_exists(filename):
#         filename = shared.filenames['rules']
#     ruleset = set()
#     for (rule,) in read_tsv_file(filename, types=(str,)):
#         ruleset.add(rule)
#     return ruleset


# TODO deprecated
# def analyze(lexicon :Lexicon, ruleset :Set[Rule], 
#             analyzer :hfst.HfstTransducer) -> None:
#     logging.getLogger('main').info('Analyzing...')
#     progressbar = tqdm.tqdm(total=len(lexicon))
#     with open_to_write(shared.filenames['analyze.graph']) as outfp:
#         for entry in lexicon.entries():
#             similar_words = set([word for word, cost in\
#                                           analyzer.lookup(entry.symstr)\
#                                           if entry.symstr < word])
#             for word in similar_words:
#                 for entry_2 in lexicon.get_by_symstr(word):
#                     if entry != entry_2:
#                         for rule in extract_all_rules(entry, entry_2):
#                             if str(rule) in ruleset:
#                                 write_line(outfp, (entry, entry_2, rule))
#             progressbar.update()
#     progressbar.close()
# 
# 
# def run():
#     lexicon = Lexicon.load(shared.filenames['analyze.wordlist'])
#     analyzer = prepare_analyzer(lexicon)
# #     rules = load_rule_set()
#     model = ModelSuite.load()
#     analyze(lexicon, rules, analyzer)

def run():
    lexicon_to_analyze = Lexicon.load(shared.filenames['analyze.wordlist'])
    analyzer = Analyzer()
    for lexitem in lexicon_to_analyze:
        analyses = analyzer.analyze(lexitem)
        for a in analyses:
#             print(a.source, a.target, a.rule, a.attr['cost'])
            print(a.source, a.target, a.rule)

