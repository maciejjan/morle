from datastruct.graph import GraphEdge
# from datastruct.lexicon import LexiconEntry
from datastruct.rules import Rule
# from models.generic import Model
# from utils.files import open_to_write, write_line
# 

from collections import defaultdict
import numpy as np
from operator import itemgetter
import scipy.sparse
from scipy.special import expit
from typing import Dict, Iterable, List, Tuple


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 10
BATCH_SIZE = 10
COEFF_SD = 0.1

# TODO currently: model AND dataset as one class; separate in the future
# TODO a logit model -- coefficients for each rule, but trained with KERAS

class LogitModel:

    def __init__(self, edges :Iterable[GraphEdge], rules :Iterable[Rule]):
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
        # TODO normalize numeric features to zero mean and unit variance?
        # -> fit feature regularization vectors B and C and the beginning
        #    X' = (X-B)*C
        #    question: globally or for each rule? -> better separately for each rule
        # -> no regularization required!
        #    - the rule coefficients show the direction of expected change
        #    - the length of the coefficient vector is the importance/certainty
        #    - the more change in this direction = the better!
        #      (but the lengths are probably not going to vary much)
        # evt. the normalized length of the change vector as a feature
        # but rescaling might be needed to ensure feature comparability
        #   and speed up convergence
        # TODO indices: object -> ID
        self.edges = edges
        self.rules = rules
        self.edge_idx = { edge: idx for idx, edge in enumerate(self.edges) }
        self.rule_idx = { rule: idx for idx, rule in enumerate(self.rules) }
        self.ngram_features = self.select_ngram_features(edges)
        # TODO features -- names of ALL features
        self.features = self.ngram_features
        self.attributes, self.selector =\
            self.extract_features_from_edges(edges)
        self.coefficients = self.initialize()
        for e in self.edges:
            print(e.source, e.target, e.rule)
        print([str(rule) for rule in self.rules])
        print(self.ngram_features)
        print(self.attributes)
        print(self.selector)
        print(self.coefficients)

    # TODO compute slice-wise

    def initialize(self):
        dim = (len(self.rules), len(self.features))
        return np.random.normal(0, COEFF_SD, dim)

    def extract_n_grams(self, word :Iterable[str]) -> Iterable[Iterable[str]]:
        result = []
        max_n = min(MAX_NGRAM_LENGTH, len(word))+1
        result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
        result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
        return result

    def select_ngram_features(self, edges :List[GraphEdge]) -> List[str]:
        # count n-gram frequencies
        ngram_freqs = defaultdict(lambda: 0)
        for edge in edges:
            for ngram in self.extract_n_grams(edge.source.symstr):
                ngram_freqs[ngram] += 1
        # select most common n-grams
        ngram_features = \
            list(map(itemgetter(0), 
                     sorted(ngram_freqs.items(), 
                            reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
        return ngram_features

    def extract_features_from_edges(self, edges :List[GraphEdge])\
                                   -> np.ndarray:
        attributes = []
        selector = scipy.sparse.dok_matrix((len(edges), len(self.rules)))
        for edge in edges:
            attribute_vec, selector_vec = [], []
            edge_ngrams = set(self.extract_n_grams(edge.source.symstr))
            for ngram in self.ngram_features:
                attribute_vec.append(1 if ngram in edge_ngrams else -1)
            selector[self.edge_idx[edge], self.rule_idx[edge.rule]] = 1
            attributes.append(attribute_vec)
        # TODO reformat attributes and selector to slices
        return np.array(attributes), selector.tocsr()

    def edge_probability(self) -> np.ndarray:
        return expit(np.sum(self.attributes *\
                            np.array(np.dot(self.selector.todense(), 
                                            self.coefficients)),
                     axis=1))

    def gradient(self, y) -> np.ndarray:
        edge_prob = self.edge_probability()
        selector = np.squeeze(np.asarray(self.selector.todense()))
        return np.dot((np.dot((y-edge_prob).reshape((-1, 1)),
                              np.ones((1, len(self.rules))))\
                       * selector).transpose(), 
                      self.attributes)

    def recompute_edge_costs(self) -> None:
        self.edge_prob = self.edge_probability()
        self.edge_costs = np.log(edge_prob) - np.log(1-edge_prob)

#     def add_rule(self, rule :Rule, domsize :int, freq :int = None) -> None:
#         super().add_rule(rule, domsize)
#         # TODO initial coefficients for the rule
#         raise NotImplementedError()
# #         if freq is not None:
# #             self.rule_features[rule][0].fit(freq)
# 
#     def fit_to_sample(self, sample :Iterable[Tuple[GraphEdge, float]]) -> None:
#         def sample_to_edges_by_rule(sample):
#             edges_by_rule = defaultdict(lambda: list())
#             for edge, weight in sample:
#                 edges_by_rule[edge.rule].append((edge, weight))
#             return edges_by_rule
# 
#         edges_by_rule = sample_to_edges_by_rule(sample)
#         # TODO fit the coefficients
#         raise NotImplementedError()
# #         for rule, edges in edges_by_rule.items():
# #             self.rule_features[rule].weighted_fit(\
# #                 self.extractor.extract_feature_values_from_weighted_edges(edges))
# 
#     def recompute_edge_costs(self, edges :Iterable[GraphEdge]) -> None:
#         for e in edges:
#             self.edge_costs[e] = self.edge_cost(e)
# 
#     def recompute_root_costs(self, roots :Iterable[LexiconEntry]) -> None:
#         for root in roots:
#             new_cost = self.rootdist.cost_of_change(\
#                     self.extractor.extract_feature_values_from_nodes([root]), [])
#             self.root_costs[root] = new_cost
# 
#     def cost_of_change(self, edges_to_add :List[GraphEdge], 
#                        edges_to_remove :List[GraphEdge]) -> float:
#         return sum(self.edge_costs[e] for e in edges_to_add) -\
#                 sum(self.root_costs[e.target] for e in edges_to_add) -\
#                 sum(self.edge_costs[e] for e in edges_to_remove) +\
#                 sum(self.root_costs[e.target] for e in edges_to_remove)
# 
#     def apply_change(self, edges_to_add :List[GraphEdge], 
#                      edges_to_remove :List[GraphEdge]) -> None:
#         self.edges_cost += sum(self.edge_costs[e] for e in edges_to_add) -\
#                            sum(self.edge_costs[e] for e in edges_to_remove)
#         self.roots_cost += sum(self.root_costs[e.target] for e in edges_to_remove) -\
#                            sum(self.root_costs[e.target] for e in edges_to_add)
# 
#     def edge_cost(self, edge):
#         return self.rule_features[edge.rule].cost(\
#             self.extractor.extract_feature_values_from_edges([edge]))
# 
#     def save_rules_to_file(self, filename :str):
#         # TODO save feature vectors
#         with open_to_write(filename) as fp:
#             for rule, features in sorted(self.rule_features.items(),
#                                          reverse=True,
#                                          key=lambda x: x[1][0].trials*x[1][0].prob):
#                 write_line(fp, (rule, features.to_string()))
# 
