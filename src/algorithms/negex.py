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
        self.lexicon_cmp_tr.minimize()
        self.lexicon_tr.convert(hfst.ImplementationType.SFST_TYPE)
        self.lexicon_cmp_tr.convert(hfst.ImplementationType.SFST_TYPE)
        self.rule_set = rule_set
        self.num_pos_ex = defaultdict(lambda: 0)
        for edge in edge_set:
            self.num_pos_ex[edge.rule] += 1
        self.rule_transducers = [rule.to_fst() for rule in rule_set]
        self.lex_transducers = [entry.to_fst() for entry in lexicon]
        for tr in self.rule_transducers:
            tr.convert(hfst.ImplementationType.SFST_TYPE)
        for tr in self.lex_transducers:
            tr.convert(hfst.ImplementationType.SFST_TYPE)
#         logging.getLogger('main').info('Precomputing some compositions...')
#         for tr in tqdm.tqdm(self.rule_transducers):
#             tr.compose(self.lexicon_cmp_tr)
#             tr.minimize()

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

    def sample_with_block_composition(self, sample_size :int):
        # TODO divide the lexicon randomly into blocks of BLOCK_SIZE words
        # TODO randomly combine a block with a rule
        # (block size = 1 => independent sample)
        # (for larger block sizes - independency sacrificed for speed)
        def _sample_block(b_id :int, r_id :int):
            tr = hfst.HfstTransducer(block_trs[b_id])
            tr.compose(self.rule_transducers[r_id])
            tr.minimize()
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

        BLOCK_SIZE = 3000

        # prepare blocks
        w_ids = list(range(len(self.lexicon)))
        random.shuffle(w_ids)
        i, j = 0, BLOCK_SIZE
        block_trs = []
        while i < len(w_ids):
            trs = [self.lex_transducers[idx] for idx in w_ids[i:j]]
            block_tr = algorithms.fst.binary_disjunct(trs)
            block_trs.append(block_tr)
            i, j = j, min(j+BLOCK_SIZE, len(w_ids))

        edge_set = EdgeSet()
        num_edges_for_rule = defaultdict(lambda: 0)
        progressbar = tqdm.tqdm(total=sample_size)
        visited = set()
        while len(edge_set) < sample_size:
            b_id = random.randrange(len(block_trs))
            r_id = random.randrange(len(self.rule_set))
            if (b_id, r_id) in visited:
                continue
            visited.add((b_id, r_id))
            rule = self.rule_set[r_id]
            for edge in _sample_block(b_id, r_id):
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

