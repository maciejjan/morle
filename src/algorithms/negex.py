import algorithms.fst
from datastruct.graph import GraphEdge, EdgeSet
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.rules import Rule, RuleSet
from utils.parallel import parallel_execute

from collections import defaultdict
import hfst
import logging
import multiprocessing
import numpy as np
from operator import itemgetter
import random
import tqdm
from typing import Callable, Iterable, List, Tuple


# TODO move to algorithms.fst
# def identity_fst():
#     tr = hfst.HfstBasicTransducer()
#     tr.set_final_weight(0, 0.0)
#     tr.add_transition(0, hfst.HfstBasicTransition(0, hfst.IDENTITY, 
#                                                   hfst.IDENTITY, 0))
#     return hfst.HfstTransducer(tr)


class NegativeExampleSampler:
    def __init__(self, lexicon :Lexicon, rule_set :RuleSet, edge_set :EdgeSet) -> None:
        self.lexicon = lexicon
#         self.lexicon_tr = lexicon_tr
#         self.lexicon_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
#         self.lexicon_cmp_tr = identity_fst()
#         self.lexicon_cmp_tr.subtract(lexicon_tr)
#         self.lexicon_cmp_tr.minimize()
#         self.lexicon_tr.convert(hfst.ImplementationType.SFST_TYPE)
#         self.lexicon_cmp_tr.convert(hfst.ImplementationType.SFST_TYPE)
        self.rule_set = rule_set
        self.num_pos_ex = defaultdict(lambda: 0)
        for edge in edge_set:
            self.num_pos_ex[edge.rule] += 1
#         self.rule_transducers = [rule.to_fst() for rule in rule_set]
#         self.lex_transducers = [entry.to_fst() for entry in lexicon]
#         for tr in self.rule_transducers:
#             tr.convert(hfst.ImplementationType.SFST_TYPE)
#             tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
#         for tr in self.lex_transducers:
#             tr.convert(hfst.ImplementationType.SFST_TYPE)

    def sample(self, sample_size :int) -> Tuple[EdgeSet, np.ndarray]:
        return self.parallel_sample_with_lookup(sample_size)

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
                         self.rule_transducers[r_id].lookup(entry.symstr)))
            if lookup_results:
                t_id = random.randrange(len(lookup_results))
                target = None
                try:
                    target = LexiconEntry(lookup_results[t_id][0])
                except Exception as e:
                    logging.getLogger('main').debug(\
                       'Exception during negative sampling: {}'.format(e))
                    continue
#                 if self.lexicon_tr.lookup(target.symstr):
                if target.symstr in self.lexicon.items_by_symstr:
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

    def parallel_sample_with_lookup(self, sample_size :int) \
                                   -> Tuple[EdgeSet, np.ndarray]:

        def _sample_process(rules :List[Rule],
                            _output_fun :Callable[..., None],
                            lexicon :Lexicon,
                            sample_size :int) -> None:
            transducers = [r.to_fst() for r in rules]
            for tr in transducers:
                tr.convert(hfst.ImplementationType.HFST_OL_TYPE)
            seen_ids = set()
            num = 0
            while num < sample_size:
                w_id = random.randrange(len(lexicon))
                r_id = random.randrange(len(rules))
                entry = lexicon[w_id]
                rule = rules[r_id]
                lookup_results = \
                    sorted(list(map(lambda x: (x[0].replace(hfst.EPSILON, ''), x[1]),
                             transducers[r_id].lookup(entry.symstr))))
                if lookup_results:
                    t_id = random.randrange(len(lookup_results))
                    if (w_id, r_id, t_id) in seen_ids:
                        continue
                    seen_ids.add((w_id, r_id, t_id))
                    target = None
                    try:
                        target = LexiconEntry(lookup_results[t_id][0])
                    except Exception as e:
                        logging.getLogger('main').debug(\
                           'Exception during negative sampling: {}'.format(e))
                        continue
#                 if self.lexicon_tr.lookup(target.symstr):
                    if target.symstr in lexicon.items_by_symstr:
                        continue
                    _output_fun(GraphEdge(entry, target, rule))
                    num += 1

        NUM_PROCESSES = 4       # TODO config parameter
        sample_size_per_proc = int(sample_size / NUM_PROCESSES)
        rules_lst = [r for r in self.rule_set]
        edges_iter = \
            parallel_execute(function=_sample_process, data=list(self.rule_set),
                             num_processes=NUM_PROCESSES,
                             additional_args=(self.lexicon,\
                                              sample_size_per_proc),
                             show_progressbar=True,
                             progressbar_total = sample_size_per_proc * NUM_PROCESSES)
        edge_set = EdgeSet()
        for edge in edges_iter:
            edge_set.add(edge)
        num_edges_for_rule = defaultdict(lambda: 0)
        for edge in edge_set:
            num_edges_for_rule[edge.rule] += 1
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

        BLOCK_SIZE = 100

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

