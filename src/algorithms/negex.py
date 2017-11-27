from datastruct.graph import GraphEdge, EdgeSet
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.rules import Rule, RuleSet
from utils.parallel import parallel_execute
import shared

from collections import defaultdict
import hfst
import logging
import numpy as np
import random
from typing import Callable, Dict, List, Tuple


class NegativeExampleSampler:
    def __init__(self, lexicon :Lexicon, rule_set :RuleSet,
                 edge_set :EdgeSet) -> None:
        self.lexicon = lexicon
        self.rule_set = rule_set
        self.num_pos_ex = defaultdict(lambda: 0)
        for edge in edge_set:
            self.num_pos_ex[edge.rule] += 1

    def sample(self, sample_size :int) -> Tuple[EdgeSet, np.ndarray]:

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
                source = lexicon[w_id]
                rule = rules[r_id]
                lookup_results = \
                    sorted(list(map(lambda x: x[0].replace(hfst.EPSILON, ''),
                                    transducers[r_id].lookup(source.symstr))))
                if lookup_results:
                    t_id = random.randrange(len(lookup_results))
                    if (w_id, r_id, t_id) in seen_ids:
                        continue
                    seen_ids.add((w_id, r_id, t_id))
                    target = None
                    try:
                        target = LexiconEntry(lookup_results[t_id])
                        if target.symstr not in lexicon.items_by_symstr:
                            _output_fun(GraphEdge(source, target, rule))
                            num += 1
                    except Exception as e:
                        logging.getLogger('main').debug(\
                           'Exception during negative sampling: {}'.format(e))

        def _compute_sample_weights(edge_set :EdgeSet) -> np.ndarray:
            'Compute weights of the sample points.'
            num_edges_for_rule = defaultdict(lambda: 0)
            for edge in edge_set:
                num_edges_for_rule[edge.rule] += 1
            result = np.empty(len(edge_set))
            for i, edge in enumerate(edge_set):
                domsize = self.rule_set.get_domsize(edge.rule)
                num_pos_ex = self.num_pos_ex[edge.rule]
                num_neg_ex = num_edges_for_rule[edge.rule]
                result[i] = (domsize-num_pos_ex) / num_neg_ex
            return result

        num_processes = shared.config['NegativeExampleSampler']\
                        .getint('num_processes')
        sample_size_per_proc = int(sample_size / num_processes)
        edges_iter = \
            parallel_execute(function=_sample_process,
                             data=list(self.rule_set),
                             num_processes=num_processes,
                             additional_args=(self.lexicon, \
                                              sample_size_per_proc),
                             show_progressbar=True,
                             progressbar_total = sample_size_per_proc * \
                                                 num_processes)
        edge_set = EdgeSet(edges_iter)
        weights = _compute_sample_weights(edge_set)
        return edge_set, weights

