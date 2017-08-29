from algorithms.align import extract_all_rules
import algorithms.fst
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.graph import GraphEdge
import shared

import hfst
import logging
from typing import List


class Analyzer:
    def __init__(self, **kwargs):
        # kwargs:
        # predict_tag :bool
        # predict_vec :bool
        self.lexicon = Lexicon.load(shared.filenames['wordlist'])
        self._compile_fst()
        self.model = ModelSuite.load()

    def analyze(self, target :LexiconEntry, **kwargs) -> List[GraphEdge]:
        # kwargs:
        # max_results :int
        # returns:
        # GraphEdge with the given target and weights
        # algorithm:
        # TODO 1a. if predict_tag: get possible tags from the tag predictor
        # TODO 1. get possible sources for the given target (self.fst.lookup)
        sources = set(sum([self.lexicon.get_by_symstr(word) \
                           for word, cost in self.fst.lookup(target.symstr)],
                          []))
        results = []
        # TODO 2. get possible (source, rule) pairs (extract rules)
        for source in sources:
            rules = extract_all_rules(source, target)
            for rule in rules:
                if rule in self.model.rule_set:
                    results.append(GraphEdge(source, target, rule))
            # TODO take only rules present in the model
        # TODO 3. rescore the analyses with the model
#         raise NotImplementedError()
#         print(results)
#         print()
        return results

    def _compile_fst(self):
        rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
        logging.getLogger('main').info('Building lexicon transducer...')
        lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
        self.fst = hfst.HfstTransducer(lexicon_tr)
        logging.getLogger('main').info('Composing with rules...')
        self.fst.compose(rules_tr)
        self.fst.minimize()
        self.fst.invert()
        self.fst.convert(hfst.ImplementationType.HFST_OLW_TYPE)

