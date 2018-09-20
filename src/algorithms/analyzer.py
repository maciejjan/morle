from algorithms.align import extract_all_rules
import algorithms.fst
from datastruct.lexicon import Lexicon, LexiconEntry, unnormalize_word
from datastruct.graph import GraphEdge, EdgeSet
from datastruct.rules import RuleSet
from models.suite import ModelSuite
from utils.files import file_exists
import shared

import hfst
import logging
import re
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
        if 'compile' not in kwargs or kwargs['compile']:
            self._compile_fst()
        self.predict_vec = 'predict_vec' in kwargs and \
                           kwargs['predict_vec'] == True
        self.enable_back_formation = 'enable_back_formation' in kwargs and \
                                     kwargs['enable_back_formation'] == True
        self.max_results = kwargs['max_results'] if 'max_results' in kwargs \
                                                 else None
        if self.predict_vec and self.enable_back_formation:
            logging.getLogger('main').warning(\
                'Vector prediction and back-formation cannot be done'
                'simultaneously. Disabling vector prediction.')
            self.predict_vec = False

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
                    if self.predict_vec:
                        target_pr = target.copy()
                        edge = GraphEdge(source, target_pr, rule)
                        target_pr.vec = self.model.predict_target_feature_vec(edge) 
                        edge_set.add(edge)
                    else:
                        edge_set.add(GraphEdge(source, target, rule))
        # back-formation
        if self.enable_back_formation and \
                (self.max_results is None or len(edge_set) < self.max_results):
            lookup_results = set()
            for w, c in self.inv_rules_tr.lookup(target.symstr):
                lookup_results.add(unnormalize_word(\
                    re.sub(hfst.EPSILON, '', w)))
            sources = [LexiconEntry(word) for word in lookup_results]
            for source in sources:
                rules = extract_all_rules(source, target)
                for rule in rules:
                    if rule in self.model.rule_set:
                        edge_set.add(GraphEdge(source, target, rule))
        if not edge_set:
            return list()
        edge_costs = self.model.edges_cost(edge_set)
        for i, edge in enumerate(edge_set):
            if edge.source not in self.lexicon:
                edge_costs[i] += self.model.root_cost(edge.source)
        results = [edge for edge in edge_set]
        for i, edge in enumerate(results):
            edge.attr['cost'] = edge_costs[i]
        results.append(GraphEdge(None, target, None, \
                                 cost=self.model.root_cost(target)))
        # 4. sort the analyses according to the cost
        results.sort(key=lambda r: r.attr['cost'])
        if self.max_results is not None:
            results = results[:self.max_results]
        return results

    def _compile_fst(self) -> None:
        rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
        self.inv_rules_tr = hfst.HfstTransducer(rules_tr)
        self.inv_rules_tr.invert()
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
        algorithms.fst.save_transducer(self.fst, filename)

    @staticmethod
    def load(filename :str, lexicon, model, **kwargs) -> None:
        kwargs['compile'] = False
        analyzer = Analyzer(lexicon, model, **kwargs)
        analyzer.fst = algorithms.fst.load_transducer(filename)
        return analyzer

