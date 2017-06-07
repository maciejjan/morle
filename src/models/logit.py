# from datastruct.graph import GraphEdge
# from datastruct.lexicon import LexiconEntry
# from datastruct.rules import Rule
# from models.generic import Model
# from utils.files import open_to_write, write_line
# 
# from collections import defaultdict
# from typing import Dict, Iterable, List, Tuple


class LogitModel:

    def __init__(self):
        self.model_type = 'logit'
#         Model.__init__(self)
#         self.edge_costs = {} # type: Dict[GraphEdge, float]
#         self.root_costs = {} # type: Dict[LexiconEntry, float]
        # TODO initialization
        # - root_features (vector -- just like a rule)
        # - rule_features: Dict[Rule, np.ndarray] (feature)
        # - edge_features: Dict[Rule, np.ndarray] (edge x feature)
        # - additional rule data (frequency, domsize)?
        # TODO extract features from edges
        # TODO cache edge feature matrices
        # TODO initialize the parameters randomly
        # TODO print edges and probabilities

    def add_rule(self, rule :Rule, domsize :int, freq :int = None) -> None:
        super().add_rule(rule, domsize)
        # TODO initial coefficients for the rule
        raise NotImplementedError()
#         if freq is not None:
#             self.rule_features[rule][0].fit(freq)

    def fit_to_sample(self, sample :Iterable[Tuple[GraphEdge, float]]) -> None:
        def sample_to_edges_by_rule(sample):
            edges_by_rule = defaultdict(lambda: list())
            for edge, weight in sample:
                edges_by_rule[edge.rule].append((edge, weight))
            return edges_by_rule

        edges_by_rule = sample_to_edges_by_rule(sample)
        # TODO fit the coefficients
        raise NotImplementedError()
#         for rule, edges in edges_by_rule.items():
#             self.rule_features[rule].weighted_fit(\
#                 self.extractor.extract_feature_values_from_weighted_edges(edges))

    def recompute_edge_costs(self, edges :Iterable[GraphEdge]) -> None:
        for e in edges:
            self.edge_costs[e] = self.edge_cost(e)

    def recompute_root_costs(self, roots :Iterable[LexiconEntry]) -> None:
        for root in roots:
            new_cost = self.rootdist.cost_of_change(\
                    self.extractor.extract_feature_values_from_nodes([root]), [])
            self.root_costs[root] = new_cost

    def cost_of_change(self, edges_to_add :List[GraphEdge], 
                       edges_to_remove :List[GraphEdge]) -> float:
        return sum(self.edge_costs[e] for e in edges_to_add) -\
                sum(self.root_costs[e.target] for e in edges_to_add) -\
                sum(self.edge_costs[e] for e in edges_to_remove) +\
                sum(self.root_costs[e.target] for e in edges_to_remove)

    def apply_change(self, edges_to_add :List[GraphEdge], 
                     edges_to_remove :List[GraphEdge]) -> None:
        self.edges_cost += sum(self.edge_costs[e] for e in edges_to_add) -\
                           sum(self.edge_costs[e] for e in edges_to_remove)
        self.roots_cost += sum(self.root_costs[e.target] for e in edges_to_remove) -\
                           sum(self.root_costs[e.target] for e in edges_to_add)

    def edge_cost(self, edge):
        return self.rule_features[edge.rule].cost(\
            self.extractor.extract_feature_values_from_edges([edge]))

    def save_rules_to_file(self, filename :str):
        # TODO save feature vectors
        with open_to_write(filename) as fp:
            for rule, features in sorted(self.rule_features.items(),
                                         reverse=True,
                                         key=lambda x: x[1][0].trials*x[1][0].prob):
                write_line(fp, (rule, features.to_string()))

