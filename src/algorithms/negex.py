import algorithms.fst
from datastruct.graph import GraphEdge, EdgeSet
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.rules import Rule, RuleSet

from collections import defaultdict
import hfst
import logging
import numpy as np
from operator import itemgetter
import random
import tqdm
from typing import Tuple


def identity_fst():
    tr = hfst.HfstBasicTransducer()
    tr.set_final_weight(0, 0.0)
    tr.add_transition(0, hfst.HfstBasicTransition(0, hfst.IDENTITY, 
                                                  hfst.IDENTITY, 0))
    return hfst.HfstTransducer(tr)


class NegativeExampleSampler:
    def __init__(self, lexicon :Lexicon, lexicon_tr :hfst.HfstTransducer,
                 rule_set :RuleSet, edge_set :EdgeSet) -> None:
        self.lexicon = lexicon
        self.lexicon_tr = lexicon_tr
#         self.lexicon_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
        self.lexicon_cmp_tr = identity_fst()
        self.lexicon_cmp_tr.subtract(lexicon_tr)
        self.lexicon_cmp_tr.convert(hfst.ImplementationType.SFST_TYPE)
        self.rule_set = rule_set
        self.num_pos_ex = defaultdict(lambda: 0)
        for edge in edge_set:
            self.num_pos_ex[edge.rule] += 1
        self.rule_transducers = [rule.to_fst() for rule in rule_set]
        self.lex_transducers = [entry.to_fst() for entry in lexicon]
#         for tr in self.transducers.values():
#             tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
        for tr in self.rule_transducers:
            tr.convert(hfst.ImplementationType.SFST_TYPE)
        for tr in self.lex_transducers:
            tr.convert(hfst.ImplementationType.SFST_TYPE)

    def sample(self, sample_size :int) -> Tuple[EdgeSet, np.ndarray]:
        return self.sample_with_block_composition(sample_size)

    # TODO works, but is heavily affected by the lookup memory leak
    # TODO start this method in a separate process to circumvent the memory leak
    def sample_with_lookup(self, sample_size :int) \
                          -> Tuple[EdgeSet, np.ndarray]:
        edge_set = EdgeSet()
        num_edges_for_rule = defaultdict(lambda: 0)
        progressbar = tqdm.tqdm(total=sample_size)
        while len(edge_set) < sample_size:
            w_id = random.randrange(len(self.lexicon))
            r_id = random.randrange(len(self.rule_set))
            entry = self.lexicon[w_id]
            rule = self.rule_set[r_id]
            lookup_results = \
                list(map(lambda x: (x[0].replace(hfst.EPSILON, ''), x[1]),
                         self.transducers[rule].lookup(entry.symstr)))
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

    def sample_with_block_composition(self, sample_size :int) -> None:
        # TODO divide the sample size into block of size, say, 100000
        # TODO sample word and rule ids
        # TODO sort the sample items by rule id
        # TODO for each rule: disjunct all input words, then compose
        #      with the rule automaton

        BLOCK_SIZE = 1000
        edge_set = EdgeSet()
        word_ids = np.array(list(range(len(self.lexicon))))

        def _sample_block(r_id :int, block_size :int) -> EdgeSet:
            w_ids = np.random.choice(word_ids, size=block_size, replace=False)
            # sort alphabetically to speedup disjunction
            w_ids_by_symstr = \
                list(map(itemgetter(0),
                         sorted([(int(w_id), self.lexicon[int(w_id)].symstr)\
                                 for w_id in w_ids],
                                key=itemgetter(1))))
            transducers = []
            tr = algorithms.fst.binary_disjunct(\
                     [hfst.HfstTransducer(self.lex_transducers[w_id]) \
                      for w_id in w_ids_by_symstr])
            tr.compose(self.rule_transducers[r_id])
            tr.compose(self.lexicon_cmp_tr)
            tr.minimize()
            result = EdgeSet()
            rule = self.rule_set[r_id]
            for input_, paths in tr.extract_paths().items():
                input_ = input_.replace(hfst.EPSILON, '')
                for entry in self.lexicon.get_by_symstr(input_):
                    for output_, weight in paths:
                        output_ = output_.replace(hfst.EPSILON, '')
                        edge = GraphEdge(entry, output_, rule)
                        if edge not in result:
                            result.add(edge)
            return result

        num_edges_for_rule = defaultdict(lambda: 0)
        progressbar = tqdm.tqdm(total=sample_size)
        r_id_queue = []
        while len(edge_set) < sample_size:
            if not r_id_queue:
                r_id_queue = list(range(len(self.rule_set)))
                random.shuffle(r_id_queue)
            r_id = r_id_queue.pop()
            rule = self.rule_set[r_id]
            for edge in _sample_block(r_id, BLOCK_SIZE):
                if edge not in edge_set:
                    edge_set.add(edge)
                    num_edges_for_rule[rule] += 1
                    progressbar.update()
                    if len(edge_set) >= sample_size:
                        break
        progressbar.close()
        # compute edge weights
        weights = np.empty(len(edge_set))
        for i, edge in enumerate(edge_set):
            domsize = self.rule_set.get_domsize(edge.rule)
            num_pos_ex = self.num_pos_ex[edge.rule]
            num_neg_ex = num_edges_for_rule[edge.rule]
            weights[i] = (domsize-num_pos_ex) / num_neg_ex
        return edge_set, weights

