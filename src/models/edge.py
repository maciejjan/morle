from datastruct.graph import GraphEdge, EdgeSet
from datastruct.rules import Rule, RuleSet
from utils.files import read_tsv_file, write_tsv_file

import numpy as np
from typing import Dict, List


class EdgeModel:
    def __init__(self, edges :List[GraphEdge], rule_domsizes :Dict[Rule, int])\
                -> None:
        raise NotImplementedError()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def null_cost(self) -> float:
        'Cost of a graph without any edges.'
        raise NotImplementedError()

    def rule_cost(self, rule :Rule) -> float:
        'Cost of having a rule in the model.'
        raise NotImplementedError()

    def recompute_costs(self) -> None:
        raise NotImplementedError()

    def fit_to_sample(self, root_weights :np.ndarray, 
                      edge_weights :np.ndarray) -> None:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'EdgeModel':
        raise NotImplementedError()


class SimpleEdgeModel(EdgeModel):
    def __init__(self, rule_set :RuleSet, alpha=1.1, beta=1.1) -> None:
#         self.edge_set = edge_set
        self.rule_set = rule_set
        self.rule_domsize = np.empty(len(rule_set))
        for i in range(len(rule_set)):
            self.rule_domsize[i] = rule_set.get_domsize(rule_set[i])
        self.alpha = alpha
        self.beta = beta
#         self.fit_to_sample(np.ones(len(edge_set)))

    def edge_cost(self, edge :GraphEdge) -> float:
        return self._rule_appl_cost[self.rule_set.get_id(edge.rule)]

    def null_cost(self) -> float:
        'Cost of a graph without any edges.'
        return self._null_cost

    def rule_cost(self, rule :Rule) -> float:
        'Cost of having a rule in the model.'
        return -self._rule_cost[self.rule_set.get_id(rule)]

#     def recompute_costs(self) -> None:
#         # no edge costs are cached, because they are readily obtained
#         # from rule costs
#         pass
# 
#     def initial_fit(self):
#         self.fit_to_sample(None, np.ones(len(self.edge_set)))

    def set_probs(self, probs :np.ndarray) -> None:
        self.rule_prob = probs
        self._rule_appl_cost = -np.log(probs) + np.log(1-probs)
        self._rule_cost = -np.log(1-probs) * self.rule_domsize
        self._null_cost = np.sum(self._rule_cost)

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        # compute rule frequencies
        rule_freq = np.zeros(len(self.rule_set))
        for i in range(weights.shape[0]):
            rule_id = self.rule_set.get_id(edge_set[i].rule)
            rule_freq[rule_id] += weights[i]
        # fit
        probs = (rule_freq + np.repeat(self.alpha-1, len(self.rule_set))) /\
                 (self.rule_domsize + np.repeat(self.alpha+self.beta-2,
                                                len(self.rule_set)))
        self.set_probs(probs)
#         self._rule_appl_cost = -np.log(self.rule_prob) +\
#                                 np.log(1-self.rule_prob)
#         self._rule_cost = -np.log(1-self.rule_prob) * self.rule_domsize
#         self._null_cost = -np.sum(self._rule_cost)

    def save(self, filename :str) -> None:
        write_tsv_file(filename, ((rule, self.rule_prob[i])\
                                  for i, rule in enumerate(self.rule_set)))

    @staticmethod
    def load(filename :str, rule_set :RuleSet) -> 'SimpleEdgeModel':
        result = SimpleEdgeModel(rule_set)
        probs = np.zeros(len(rule_set))
        for rule, prob in read_tsv_file(filename, (str, float)):
            probs[rule_set.get_id(rule_set[rule])] = prob
        result.set_probs(probs)
        return result


# TODO sampling of negative examples:
# use the automaton (L .o. R - L .o. R .o. L), not necessarily minimized
# also: try passing alignments to a RNN instead of rule embedding
class NeuralEdgeModel(EdgeModel):
    pass


# TODO also using sampling of negative examples
class LogisticEdgeModel(EdgeModel):
    pass


# TODO pass alignments on character level to an RNN instead of rule embedding
class AlignmentRNNEdgeModel(EdgeModel):
    pass

