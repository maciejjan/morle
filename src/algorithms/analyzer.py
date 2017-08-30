from algorithms.align import extract_all_rules
import algorithms.fst
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.graph import GraphEdge
from datastruct.rules import RuleSet
from models.neural import ModelSuite
from utils.files import file_exists
import shared

import hfst
import logging
from typing import List


class Analyzer:
    def __init__(self, **kwargs):
        # kwargs:
        # predict_tag :bool
        # predict_vec :bool
        # TODO pass those things as parameters rather than loading them here!!!
        self.lexicon = Lexicon.load(shared.filenames['wordlist'])
        rules_file = shared.filenames['rules-modsel']
        if not file_exists(rules_file):
            rules_file = shared.filenames['rules']
        self.rule_set = RuleSet.load(rules_file)
        logging.getLogger('main').info('Loaded %d rules.' % len(self.rule_set))
#         edge_set = EdgeSet.load(shared.filenames['graph', lexicon, rule_set)
        # TODO empty edge set???
        self.model = ModelSuite.load()
        self._compile_fst()

    def analyze(self, target :LexiconEntry, **kwargs) -> List[GraphEdge]:
        # kwargs:
        # max_results :int
        # returns:
        # GraphEdge with the given target and weights
        # algorithm:
        # TODO 1a. if predict_tag: get possible tags from the tag predictor
        # 1. get possible sources for the given target
        sources = set(sum([self.lexicon.get_by_symstr(word) \
                           for word, cost in self.fst.lookup(target.symstr)],
                          []))
        result_set = EdgeSet()
        # 2. get possible (source, rule) pairs (extract rules)
        for source in sources:
            rules = extract_all_rules(source, target)
            for rule in rules:
                if rule in self.rule_set:
                    result_set.add(GraphEdge(source, target, rule))
        # 3. rescore the analyses using the model
        costs = self.model.edges_cost(result_set)
        results = []
        for i, edge in enumerate(result_set):
            edge.attr['cost'] = costs[i]
            results.append(edge)
        results.sort(key=lambda r: r.attr['cost'])
        # 4. sort the analyses according to the cost
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

