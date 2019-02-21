#TODO deprecated module

import algorithms.fst
from datastruct.lexicon import normalize_word, unnormalize_word
from datastruct.rules import Rule
from utils.files import file_exists, open_to_write, read_tsv_file, write_line
import shared

import hfst
import logging
import tqdm
from typing import List


def load_rules() -> List[Rule]:
    filename = shared.filenames['rules-fit']
    if not file_exists(filename):
        filename = shared.filenames['rules-modsel']
    if not file_exists(filename):
        filename = shared.filenames['rules']
    rules = []  # type: List[Rule]
    for (rule_str,) in read_tsv_file(filename, types=(str,)):
        rules.append(Rule.from_string(rule_str))
    return rules


def build_rule_transducer(rules :List[Rule]) -> hfst.HfstTransducer:
    return algorithms.fst.binary_disjunct(\
               [rule.to_fst() for rule in rules],
               print_progress=True)


def run():
#     lexicon = Lexicon(filename=shared.filenames['wordlist'])
#     to_analyze = Lexicon(filename=shared.filenames['analyze.wordlist'])
    analyzer = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
#     analyzer.compose(algorithms.fst.load_transducer(shared.filenames['rules-tr']))
    analyzer.compose(build_rule_transducer(load_rules()))
    analyzer.minimize()
    analyzer.invert()
    analyzer.convert(hfst.ImplementationType.HFST_OLW_TYPE)

    with open_to_write(shared.filenames['analyze.graph']) as outfp:
        # TODO normalize and unnormalize
        for (word,) in read_tsv_file(shared.filenames['analyze.wordlist'],
                                     (str,), show_progressbar=True):
            try:
                word_norm = normalize_word(word)
                similar_words = set([w for w, c in analyzer.lookup(word_norm)])
                for w in similar_words:
                    write_line(outfp, (unnormalize_word(w), word))
            except Exception as e:
                logging.getLogger('main').warning('ignoring %s: %s' %\
                                                  (word, str(e)))

