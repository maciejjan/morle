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


class NegativeExampleSampler:
    def __init__(self, lexicon :Lexicon, lexicon_tr :hfst.HfstTransducer,
                 rule_set :RuleSet, edge_set :EdgeSet) -> None:
        self.lexicon = lexicon
        self.lexicon_tr = lexicon_tr
        self.lexicon_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
        self.rule_set = rule_set
        self.num_pos_ex = defaultdict(lambda: 0)
        for edge in edge_set:
            self.num_pos_ex[edge.rule] += 1
        self.transducers = { rule : rule.to_fst() for rule in rule_set }
        for tr in self.transducers.values():
            tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)

    # TODO works, but is heavily affected by the lookup memory leak
    # TODO start this method in a separate process to circumvent the memory leak
    def sample(self, sample_size :int) -> Tuple[EdgeSet, np.ndarray]:
        edge_set = EdgeSet()
        num_edges_for_rule = defaultdict(lambda: 0)
        progressbar = tqdm.tqdm(total=sample_size)
        while len(edge_set) < sample_size:
            w_id = random.randrange(len(self.lexicon))
            r_id = random.randrange(len(self.rule_set))
            entry = self.lexicon[w_id]
            rule = self.rule_set[r_id]
            lookup_results = self.transducers[rule].lookup(entry.symstr)
            if lookup_results:
                t_id = random.randrange(len(lookup_results))
                target = None
                try:
                    target = LexiconEntry(lookup_results[t_id][0])
                except Exception as e:
                    logging.getLogger('main').debug(\
                       'Exception during negative sampling: {}'.format(e))
                    continue
                if self.lexicon_tr.lookup(target.symstr):
                    continue
                edge = GraphEdge(entry, target, rule)
                if edge not in edge_set:
                    edge_set.add(edge)
                    num_edges_for_rule[rule] += 1
                    progressbar.update()
        progressbar.close()
        # compute edge weights
        weights = np.empty(len(edge_set))
        for i, edge in enumerate(edge_set):
            domsize = self.rule_set.get_domsize(edge.rule)
            num_pos_ex = self.num_pos_ex[edge.rule]
            num_neg_ex = num_edges_for_rule[edge.rule]
            weights[i] = (domsize-num_pos_ex) / num_neg_ex
        return edge_set, weights

    # TODO performs the sampling in a separate process to circumvent the memory leak
    def sample_spawned(self, sample_size :int) -> Tuple[EdgeSet, np.ndarray]:
        raise NotImplementedError()

