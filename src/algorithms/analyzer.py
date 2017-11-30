from algorithms.align import extract_all_rules
import algorithms.fst
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.graph import GraphEdge, EdgeSet
from datastruct.rules import RuleSet
from models.suite import ModelSuite
from utils.files import file_exists
import shared

import hfst
import logging
from typing import List

# TODO special cases:
# - no feature model -> quicker analysis with compose + minimize + lookup
# - tag prediction

# TODO
# - predict feature fector if not present

class Analyzer:
    def __init__(self, lexicon :Lexicon, model :ModelSuite, **kwargs):
        # kwargs:
        # predict_tag :bool
        # predict_vec :bool
        # TODO pass those things as parameters rather than loading them here!!!
        self.lexicon = lexicon
        self.model = model
        self._compile_fst()

    def analyze(self, target :LexiconEntry, **kwargs) -> List[GraphEdge]:
        # TODO 1a. if predict_tag: get possible tags from the tag predictor
        # 1. get possible sources for the given target
        sources = set(sum([self.lexicon.get_by_symstr(word) \
                           for word, cost in self.fst.lookup(target.symstr)],
                          []))
        results = []
        # 2. get possible (source, rule) pairs (extract rules) and score them
        edge_set = EdgeSet(self.lexicon)
        for source in sources:
            rules = extract_all_rules(source, target)
            for rule in rules:
                if rule in self.model.rule_set:
#                     edge.attr['cost'] = self.model.edges_cost(edge)
#                     results.append(GraphEdge(source, target, rule))
                    edge_set.add(GraphEdge(source, target, rule))
        if not edge_set:
            return list()
        edge_costs = self.model.edges_cost(edge_set)
        results = [edge for edge in edge_set]
        for i, edge in enumerate(results):
            edge.attr['cost'] = edge_costs[i]
        results.sort(key=lambda r: r.attr['cost'])
        # 4. sort the analyses according to the cost
        # (TODO max_results etc.)
        return results

    def _compile_fst(self) -> None:
        rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
        logging.getLogger('main').info('Building lexicon transducer...')
        lexicon_tr = algorithms.fst.load_transducer(\
                       shared.filenames['lexicon-tr'])
        self.fst = hfst.HfstTransducer(lexicon_tr)
        logging.getLogger('main').info('Composing with rules...')
        self.fst.compose(rules_tr)
        self.fst.minimize()
        self.fst.invert()
        self.fst.convert(hfst.ImplementationType.HFST_OLW_TYPE)

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> None:
        raise NotImplementedError()

