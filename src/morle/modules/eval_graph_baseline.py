import algorithms.fst
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
from datastruct.graph import FullGraph
from utils.files import read_tsv_file
import shared

import hfst
import logging
from typing import Set
import tqdm


def build_unweighted_rule_transducer(filename :str) -> hfst.HfstTransducer:
    transducers = []
    for (rule_str,) in read_tsv_file(filename, (str,)):
        transducers.append(Rule.from_string(rule_str).to_fst())
    return algorithms.fst.binary_disjunct(transducers, print_progress=True)


def prepare_analyzer(lexicon :Lexicon, rules_tr :hfst.HfstTransducer) \
                    -> hfst.HfstTransducer:
    logging.getLogger('main').info('Building lexicon transducer...')
#     lexicon_2 = Lexicon()
#     for entry in list(lexicon.entries())[:1000]:
#         lexicon_2.add(entry)
#     lexicon_tr = lexicon_2.to_fst()
    lexicon_tr = lexicon.to_fst()
    analyzer = hfst.HfstTransducer(lexicon_tr)
    logging.getLogger('main').info('Composing with rules...')
    analyzer.compose(rules_tr)
    analyzer.minimize()
    logging.getLogger('main').info('Composing again with lexicon...')
    analyzer.compose(lexicon_tr)
    analyzer.minimize()
    analyzer.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    return analyzer


def run():
    lexicon = Lexicon(filename=shared.filenames['analyze.wordlist'])
    rules_tr = build_unweighted_rule_transducer(shared.filenames['rules'])
    analyzer = prepare_analyzer(lexicon, rules_tr)

    # load evaluation data
    eval_edges = set()
    for word_1, word_2 in read_tsv_file(shared.filenames['eval.graph']):
        eval_edges.add((word_1, word_2))

    logging.getLogger('main').info('Analyzing...')
    progressbar = tqdm.tqdm(total=len(lexicon))
    edges = set()
    for entry in lexicon.entries():
        similar_words = set([word for word, cost in\
                                      analyzer.lookup(entry.symstr)\
                                      if entry.symstr < word])
        for word in similar_words:
            for entry_2 in lexicon.get_by_symstr(word):
                edges.add((str(entry), str(entry_2)))
        progressbar.update()
    progressbar.close()

    tp = len(edges & eval_edges)
    fp = len(edges - eval_edges)
    fn = len(eval_edges - edges)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fscore = 2/((1/precision)+(1/recall))
    print('TP:', tp)
    print('FP:', fp)
    print('FN:', fn)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F-score:', fscore)

