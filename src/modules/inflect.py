from algorithms.align import extract_all_rules
import algorithms.fst
from datastruct.lexicon import tokenize_word, normalize_seq, normalize_word, \
                               unnormalize_word, Lexicon, LexiconEntry
from datastruct.graph import GraphEdge
from models.suite import ModelSuite
from utils.files import read_tsv_file
import shared

import hfst
import logging
from operator import itemgetter
import sys
import tqdm


# TODO refactor
def run():

    def _extract_tag(word):
        return ''.join(tokenize_word(word)[1])

    lexicon = Lexicon.load(shared.filenames['wordlist'])
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    rules_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    alphabet = lexicon_tr.get_alphabet()
    model = ModelSuite.load()
    max_results = shared.config['inflect'].getint('max_results')
    pairs = []
    # FIXME is there a better solution for creating lists of LexiconEntry
    # objects and skipping the ones for which exceptions are thrown?
    for lemma, tag in read_tsv_file(shared.filenames['analyze.wordlist']):
        try:
            pairs.append((LexiconEntry(lemma), tag))
        except Exception as e:
            logging.warning(e)
    for lemma, tag in tqdm.tqdm(pairs):
        try:
            lookup_results = rules_tr.lookup(lemma.symstr)
            inflections = []
            for w, c in lookup_results:
                if _extract_tag(w) == tag:
                    try:
                        inflections.append(LexiconEntry(unnormalize_word(w)))
                    except Exception as e:
                        logging.getLogger('main').warning(e)
            edges = []
            for infl in inflections:
                for rule in extract_all_rules(lemma, infl):
                    if rule in model.rule_set:
                        edge = GraphEdge(lemma, infl, rule)
                        edge.attr['cost'] = model.edge_cost(edge)
                        edges.append(edge)
            edges = sorted(edges, key=lambda x: x.attr['cost'])[:max_results]
            for e in edges:
                print(lemma, e.target, e.attr['cost'], sep='\t')
            if not edges:
                print(lemma, '---'+tag, '---', sep='\t')
        except Exception as e:
            logging.getLogger('main').warning(e)
            print(lemma, '---'+tag, '---', sep='\t')

