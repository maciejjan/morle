from datastruct.graph import GraphEdge, EdgeSet
from datastruct.rules import Rule, RuleSet
from utils.files import read_tsv_file, write_tsv_file

from keras.models import Model
from keras.layers import concatenate, Dense, Embedding, Flatten, Input
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
        self.rule_set = rule_set
        self.rule_domsize = np.empty(len(rule_set))
        for i in range(len(rule_set)):
            self.rule_domsize[i] = rule_set.get_domsize(rule_set[i])
        self.alpha = alpha
        self.beta = beta

    def edge_cost(self, edge :GraphEdge) -> float:
        return self._rule_appl_cost[self.rule_set.get_id(edge.rule)]

    def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
        result = np.zeros(len(edge_set))
        for rule, edge_ids in edge_set.get_edge_ids_by_rule().items():
            result[edge_ids] = self._rule_appl_cost[self.rule_set.get_id(rule)]
        return result

    def null_cost(self) -> float:
        'Cost of a graph without any edges.'
        return self._null_cost

    def rule_cost(self, rule :Rule) -> float:
        'Cost of having a rule in the model.'
        return -self._rule_cost[self.rule_set.get_id(rule)]

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


class NGramFeatureExtractor:
    def __init__(self) -> None:
        raise NotImplementedError()

    def select_features(self, edge_set :EdgeSet) -> None:
        '''Count n-grams and select the most frequent ones.'''
        raise NotImplementedError()

    def extract(self, edge_set :EdgeSet) -> np.ndarray:
        '''Extract n-gram features from edges and return a binary matrix.'''
        raise NotImplementedError()


class NeuralEdgeModel(EdgeModel):
    def __init__(self, rule_set :RuleSet, edge_set :EdgeSet,
                       negex_sampler :NegativeExampleSampler) -> None:
        self.rule_set = rule_set
        self.negex_sampler = negex_sampler
        self.ngram_extractor = NGramFeatureExtractor()
        self.ngram_extractor.select_features(edge_set)
        self._compile_network()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
        raise NotImplementedError()

    def null_cost(self, edge_set :EdgeSet) -> float:
        'Cost of a graph without any edges.'
        # TODO log(1-prob) for all edges in edge_set
        # TODO additionally: weight*log(1-prob) for sampled negative examples
        raise NotImplementedError()

    def rule_cost(self, rule :Rule) -> float:
        'Cost of having a rule in the model.'
        raise NotImplementedError()

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str, rule_set :RuleSet) -> 'SimpleEdgeModel':
        raise NotImplementedError()

    def _compile_network(self):
        num_ngr = self.ngram_extractor.num_features()
        input_attr = Input(shape=(num_ngr,), name='input_attr')
        input_rule = Input(shape=(1,), name='input_rule')
        rule_emb = Embedding(input_dim=num_rules, output_dim=30,\
                             input_length=1)(input_rule)
        rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
        concat = concatenate([input_attr, rule_emb_fl])
        internal = Dense(30, activation='relu', name='internal')(concat)
        output = Dense(1, activation='sigmoid', name='dense')(internal)
        self.nn = Model(inputs=[input_attr, input_rule], outputs=[output])
        self.nn.compile(optimizer='adam', loss='mse')

    def _prepare_data(self, edge_set :EdgeSet) -> \
                     Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()
#         # build the arrays as lists
#         X_attr_lst, X_rule_lst, y_lst = [], [], []
#         for edge in edge_set:
#             X_attr_lst.append(edge.source.vec)
#             X_rule_lst.append(self.rule_set.get_id(edge.rule))
#             y_lst.append(edge.target.vec)
#         # convert the lists into matrices
#         X_attr = np.vstack(X_attr_lst)
#         X_rule = np.array(X_rule_lst)
#         y = np.vstack(y_lst)
#         return X_attr, X_rule, y


# TODO also using sampling of negative examples
class LogisticEdgeModel(EdgeModel):
    pass


# TODO pass alignments on character level to an RNN instead of rule embedding
class AlignmentRNNEdgeModel(EdgeModel):
    pass

