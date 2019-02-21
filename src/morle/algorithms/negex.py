from morle.datastruct.graph import GraphEdge, EdgeSet
from morle.datastruct.lexicon import LexiconEntry, Lexicon
from morle.datastruct.rules import Rule, RuleSet
from morle.utils.parallel import parallel_execute
import morle.shared as shared

from collections import defaultdict
import hfst
import logging
import numpy as np
import random
from typing import Callable, Dict, List, Tuple


class NegativeExampleSampler:
    def __init__(self, rule_set :RuleSet) -> None:
        self.rule_set = rule_set

    def sample(self, lexicon :Lexicon, sample_size :int,
               show_progressbar :bool = True) -> EdgeSet:

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

        num_processes = shared.config['NegativeExampleSampler']\
                        .getint('num_processes')
        sample_size_per_proc = int(sample_size / num_processes)
        edges_iter = \
            parallel_execute(function=_sample_process,
                             data=list(self.rule_set),
                             num_processes=num_processes,
                             additional_args=(lexicon, sample_size_per_proc),
                             show_progressbar=show_progressbar,
                             progressbar_total = sample_size_per_proc * \
                                                 num_processes)
        edge_set = EdgeSet(lexicon, edges_iter)
        return edge_set

    def compute_sample_weights(self, sample_edges :EdgeSet,
                               positive_edges :EdgeSet) -> np.ndarray:
        'Compute weights of the sample points.'
        num_positive_edges_for_rule = np.zeros(len(self.rule_set))
        for edge in positive_edges:
            num_positive_edges_for_rule[self.rule_set.get_id(edge.rule)] += 1
        num_sample_edges_for_rule = np.zeros(len(self.rule_set))
        for edge in sample_edges:
            num_sample_edges_for_rule[self.rule_set.get_id(edge.rule)] += 1
        result = np.empty(len(sample_edges))
        for i, edge in enumerate(sample_edges):
            domsize = self.rule_set.get_domsize(edge.rule)
            r_id = self.rule_set.get_id(edge.rule)
            num_pos_ex = num_positive_edges_for_rule[r_id]
            num_neg_ex = num_sample_edges_for_rule[r_id]
            result[i] = (domsize-num_pos_ex) / num_neg_ex
        return result

