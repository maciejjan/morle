import algorithms.fst
from datastruct.lexicon import LexiconEntry
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
        self.fst = self._compile_fst()
#         self.model = ModelSuite.load()

    def analyze(target :LexiconEntry, **kwargs) -> List[GraphEdge]:
        # kwargs:
        # max_results :int
        # returns:
        # GraphEdge with the given target and weights
        # algorithm:
        # 1a. if predict_tag: get possible tags from the tag predictor
        # 1. get possible sources for the given target (self.fst.lookup)
        # 2. get possible (source, rule) pairs (extract rules)
        # 3. rescore the analyses with the model
        raise NotImplementedError()

    def _compile_fst(self):
        rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
        logging.getLogger('main').info('Building lexicon transducer...')
        lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
        self.fst = hfst.HfstTransducer(lexicon_tr)
        logging.getLogger('main').info('Composing with rules...')
        self.fst.compose(rules_tr)
        self.fst.minimize()
        self.fst.convert(hfst.ImplementationType.HFST_OLW_TYPE)

