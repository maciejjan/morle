from datastruct.graph import GraphEdge

from typing import List, Tuple


class BaselineModel:
    # TODO: fitting the rule probabilities to the average frequencies
    def __init__(self, edges :List[GraphEdge]):
        self.rule_prob = {}
        for edge in edges:
            if edge.rule not in self.rule_prob:
                self.rule_prob[edge.rule] = 0.5

    def fit_to_sample(self, edges_freq :List[Tuple[GraphEdge, float]]) -> None:
        rule_prob_sums, rule_num_edges = {}, {}
        for edge, prob in edges_freq:
            if edge.rule not in rule_prob_sums:
                rule_prob_sums[edge.rule] = 0.0
                rule_num_edges[edge.rule] = 0
            rule_prob_sums[edge.rule] += prob
            rule_num_edges[edge.rule] += 1
        for rule in rule_prob_sums:
            self.rule_prob[rule] = rule_prob_sums[rule] / rule_num_edges[rule]

    def recompute_edge_prob(self) -> None:
        pass

    def edge_prob(self, edge :GraphEdge) -> float:
        return self.rule_prob[edge.rule]

