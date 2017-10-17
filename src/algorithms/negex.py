import algorithms.fst
from datastruct.graph import GraphEdge, EdgeSet
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.rules import RuleSet

from collections import defaultdict
import hfst
import logging
import numpy as np
import random
import tqdm
from typing import Tuple


# TODO deprecated
# def identity_fst() -> hfst.HfstTransducer:
#     tr = hfst.HfstBasicTransducer()
#     tr.add_transition(0, hfst.HfstBasicTransition(0, hfst.IDENTITY,
#                                                   hfst.IDENTITY, 0.0))
#     tr.set_final_weight(0, 0.0)
#     return hfst.HfstTransducer(tr)


class NegativeExampleSampler:
    # TODO this is the natural place to store domsizes
    # TODO sample examples for each rule separately
    # TODO sample for each rule as many negative examples
    #      as there are edges with this rule (= potential positive examples)
    # TODO stores also weights of sample items (domsize/sample_size for each rule)

    def __init__(self, lexicon :Lexicon, lexicon_tr :hfst.HfstTransducer,
                 rule_set :RuleSet, edge_set :EdgeSet) -> None:
        self.lexicon = lexicon
        self.lexicon_tr = lexicon_tr
        self.lexicon_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
        self.rule_set = rule_set
        self.num_pos_ex = defaultdict(lambda: 0)
        for edge in edge_set:
            self.num_pos_ex[edge.rule] += 1
#         self.non_lex_tr = identity_fst()
#         self.non_lex_tr.subtract(self.lexicon_tr)
#         self.rule_example_counts = rule_example_counts
#         self.rule_domsizes = rule_domsizes
        self.transducers = { rule : rule.to_fst() for rule in rule_set }
        for tr in self.transducers.values():
            tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
#         self.transducers = self._build_transducers(lexicon_tr)

    # TODO works, but is heavily affected by the lookup memory leak
    # TODO start this method in a separate process to circumvent the memory leak
    def sample(self, sample_size :int) -> Tuple[EdgeSet, np.ndarray]:
        edge_set = EdgeSet()
        num_edges_for_rule = defaultdict(lambda: 0)
#         visited_ids = set()
        progressbar = tqdm.tqdm(total=sample_size)
        while len(edge_set) < sample_size:
            w_id = random.randrange(len(self.lexicon))
            r_id = random.randrange(len(self.rule_set))
            entry = self.lexicon[w_id]
            rule = self.rule_set[r_id]
            lookup_results = self.transducers[rule].lookup(entry.symstr)
            if lookup_results:
                t_id = random.randrange(len(lookup_results))
#                 if (w_id, r_id, t_id) in visited_ids:
#                     continue
#                 visited_ids.add((w_id, r_id, t_id))
                target = None
                try:
                    target = LexiconEntry(lookup_results[t_id][0])
                except Exception:
                    continue
                if self.lexicon_tr.lookup(target.symstr):
                    continue
                edge = GraphEdge(entry, target, rule)
                if edge not in edge_set:
                    edge_set.add(edge)
                    num_edges_for_rule[rule] += 1
                    progressbar.update()
#                 else:
#                     print(*edge.to_tuple(), 'already in EdgeSet')
        progressbar.close()
        # compute edge weights
        weights = np.empty(len(edge_set))
        for i, edge in enumerate(edge_set):
            domsize = self.rule_set.get_domsize(edge.rule)
            num_pos_ex = self.num_pos_ex[edge.rule]
            num_neg_ex = num_edges_for_rule[edge.rule]
            weights[i] = (domsize-num_pos_ex) / num_neg_ex
        return edge_set, weights

# TODO deprecated
#     def _build_transducers(self, lexicon_tr :hfst.HfstTransducer):
#         result = {}
#         non_lex_tr = identity_fst()
#         non_lex_tr.subtract(lexicon_tr)
#         logging.getLogger('main').info('Building transducers for negative sampling...')
#         for rule in tqdm.tqdm(self.rule_set):
#             tr = hfst.HfstTransducer(lexicon_tr)
#             tr.compose(rule.to_fst())
#             tr.minimize()
#             tr.compose(non_lex_tr)
#             tr.minimize()
#             result[rule] = tr
#         return result

