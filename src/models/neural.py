import algorithms.alergia
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.graph import GraphEdge, EdgeSet, FullGraph, Branching
from datastruct.rules import Rule, RuleSet
from models.generic import Model
from utils.files import open_to_write, write_line, write_tsv_file
import shared

from collections import defaultdict
import hfst
import math
import numpy as np
from operator import itemgetter
from scipy.stats import multivariate_normal
from typing import Dict, Iterable, List, Tuple
import sys

import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 50
MAX_NEGATIVE_EXAMPLES = 1000000

# TODO currently: model AND dataset as one class; separate in the future

# TODO further ideas:
# - take also n-grams of the target word -- useful for e.g. insertion rules
# - take also n-grams around alternation spots

class RootModel:
    def __init__(self, entries :Iterable[LexiconEntry]) -> None:
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'RootModel':
        raise NotImplementedError()


class AlergiaRootModel(RootModel):

    def __init__(self, lexicon :Lexicon = None) -> None:
        self.lexicon = lexicon
        if self.lexicon is None:
            self.automaton = hfst.empty_fst()
        else:
            self.fit()

    def fit(self) -> None:
        word_seqs, tag_seqs = [], []
        for entry in self.lexicon:
            word_seqs.append(entry.word)
            tag_seqs.append(entry.tag)

        alpha = shared.config['compile'].getfloat('alergia_alpha')
        freq_threshold = \
            shared.config['compile'].getint('alergia_freq_threshold')
        self.automaton = \
            algorithms.alergia.alergia(word_seqs, alpha=alpha, 
                                       freq_threshold=freq_threshold).to_hfst()
        tag_automaton = \
            algorithms.alergia.prefix_tree_acceptor(tag_seqs).to_hfst()
        tag_automaton.minimize()

        self.automaton.concatenate(tag_automaton)
        self.automaton.remove_epsilons()
        self.automaton.convert(hfst.ImplementationType.HFST_OLW_TYPE)
        self.recompute_costs()
            
    def recompute_costs(self) -> None:
        self.costs = np.empty(len(self.lexicon))
        for i, entry in enumerate(self.lexicon):
            self.costs[i] = self.automaton.lookup(entry.symstr)[0][1]

    def root_cost(self, entry :LexiconEntry) -> float:
        return self.costs[self.lexicon.get_id(entry)]

    def save(self, filename :str) -> None:
        algorithms.fst.save_transducer(self.automaton, filename)

    @staticmethod
    def load(filename :str, lexicon :Lexicon) -> 'AlergiaRootModel':
        result = AlergiaRootModel()
        result.lexicon = lexicon
        result.automaton = algorithms.fst.load_transducer(filename)
        result.recompute_costs()
        return result


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

    def fit_to_sample(self, sample :np.ndarray) -> None:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'EdgeModel':
        raise NotImplementedError()


class BernoulliEdgeModel(EdgeModel):
    def __init__(self, edge_set :EdgeSet, rule_set :RuleSet,
                 alpha=1.1, beta=1.1) -> None:
        self.edge_set = edge_set
        self.rule_set = rule_set
        self.rule_domsize = np.empty(len(rule_set))
        for i in range(len(rule_set)):
            self.rule_domsize[i] = rule_set.get_domsize(rule_set[i])
        # actually alpha-1 and beta-1: the prior hyperparameters
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

    def recompute_costs(self) -> None:
        # no edge costs are cached, because they are readily obtained
        # from rule costs
        pass

    def fit_to_sample(self, sample :np.ndarray) -> None:
        # compute rule frequencies
        rule_freq = np.zeros(len(self.rule_set))
        for i in range(sample.shape[0]):
            rule_id = self.rule_set.get_id(self.edge_set[i].rule)
            rule_freq[rule_id] += sample[i]
        # fit
        self.rule_prob = \
            (rule_freq + np.repeat(self.alpha-1, len(self.rule_set))) /\
            (self.rule_domsize + np.repeat(self.alpha+self.beta-2,
                                           len(self.rule_set)))
        self._rule_appl_cost = -np.log(self.rule_prob) +\
                                np.log(1-self.rule_prob)
        self._rule_cost = -np.log(1-self.rule_prob) * self.rule_domsize
        self._null_cost = -np.sum(self._rule_cost)

    def save(self, filename :str) -> None:
        write_tsv_file(filename, ((rule, self.rule_prob[i])\
                                  for i, rule in enumerate(self.rule_set)))

    @staticmethod
    def load(filename :str) -> 'BernoulliEdgeModel':
        raise NotImplementedError()


class NeuralEdgeModel(EdgeModel):
    pass


class FeatureModel:
    def __init__(self, graph :FullGraph) -> None:
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def recompute_costs(self) -> None:
        raise NotImplementedError()

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'FeatureModel':
        raise NotImplementedError()


class NeuralFeatureModel(FeatureModel):
    def __init__(self, graph :FullGraph) -> None:
        self._prepare_data(graph)
        self._compile_model()
        self.model.fit([self.X_attr, self.X_rule], self.y, epochs=10,
                       batch_size=1000, verbose=1)
        self.recompute_costs()
        self.save_costs_to_file('costs-neural.txt')

    def root_cost(self, entry :LexiconEntry) -> float:
        return float(self.costs[self.word_idx[entry]])

    def edge_cost(self, edge :GraphEdge) -> float:
        return float(self.costs[self.edge_idx[edge]])

    def recompute_costs(self) -> None:
        self.y_pred = self.model.predict([self.X_attr, self.X_rule])
        self._fit_error()
        error = self.y - self.y_pred
        self.costs = \
            -multivariate_normal.logpdf(error, np.zeros(self.y.shape[1]),
                                        self.error_cov)

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        weights = np.empty(self.y.shape[0])
        for idx in self.word_idx.values():
            weights[idx] = 1.0
        for edge, weight in sample:
            weights[self.edge_idx[edge]] = weight
            weights[self.word_idx[edge.target]] -= weight
        self.model.fit([self.X_attr, self.X_rule], self.y, 
                       epochs=10, sample_weight=weights,
                       batch_size=1000, verbose=1)

    def save_costs_to_file(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for entry, idx in sorted(self.word_idx.items(), key=itemgetter(1)):
                write_line(fp, (str(entry), self.root_cost(entry)))
            for edge, idx in sorted(self.edge_idx.items(), key=itemgetter(1)):
                edge_cost = self.edge_cost(edge)
                edge_gain = edge_cost - self.root_cost(edge.target)
                write_line(fp, (str(edge.source), str(edge.target),
                                str(edge.rule), edge_cost, edge_gain))

    def _prepare_data(self, graph :FullGraph):
        self.ngram_features = self._select_ngram_features(graph.nodes_iter())
        ngram_features_hash = {}
        for i, ngram in enumerate(self.ngram_features):
            ngram_features_hash[ngram] = i
        self.word_idx = { entry : idx for idx, entry in enumerate(graph.nodes_iter()) }
        self.edge_idx = { edge : idx for idx, edge in \
                                enumerate(graph.iter_edges(), len(self.word_idx)) }
        self.rule_idx = { rule : idx for idx, rule in \
                                enumerate(set(edge.rule for edge in self.edge_idx), 1) }
        vector_dim = shared.config['Features'].getint('word_vec_dim')
        sample_size = len(self.word_idx) + len(self.edge_idx)
        num_features = len(self.ngram_features) +\
                       shared.config['Features'].getint('word_vec_dim')
        self.X_attr = np.zeros((sample_size, num_features))
        self.X_rule = np.empty(sample_size)
        self.y = np.empty((sample_size, vector_dim))
        for entry, idx in self.word_idx.items():
            for ngram in self._extract_n_grams(entry.word + entry.tag):
                if ngram in ngram_features_hash:
                    self.X_attr[idx, ngram_features_hash[ngram]] = 1
            self.X_rule[idx] = 0
            self.y[idx] = entry.vec
        for edge, idx in self.edge_idx.items():
            for ngram in self._extract_n_grams(edge.source.word + edge.source.tag):
                if ngram in ngram_features_hash:
                    self.X_attr[idx, ngram_features_hash[ngram]] = 1
            self.X_attr[idx, len(ngram_features_hash):] = edge.source.vec
            self.X_rule[idx] = self.rule_idx[edge.rule]
            self.y[idx] = edge.target.vec

    def _compile_model(self) -> None:
        vector_dim = shared.config['Features'].getint('word_vec_dim')
        num_rules = len(self.rule_idx)+1
        num_features = len(self.ngram_features) + vector_dim
        input_attr = Input(shape=(num_features,), name='input_attr')
        dense_attr = Dense(100, activation='softplus', name='dense_attr')\
                     (input_attr)
        input_rule = Input(shape=(1,), name='input_rule')
        rule_emb = Embedding(input_dim=num_rules, output_dim=100,\
                             input_length=1)(input_rule)
        rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
        concat = keras.layers.concatenate([dense_attr, rule_emb_fl])
        output = Dense(vector_dim, activation='linear', name='dense')(concat)

        self.model = Model(inputs=[input_attr, input_rule], outputs=[output])
        self.model.compile(optimizer='adam', loss='mse')

    def _fit_error(self):
        n = self.y.shape[0]
        error = self.y - self.y_pred
        self.error_cov = np.dot(error.T, error)/n

    def _select_ngram_features(self, entries :Iterable[LexiconEntry]) -> List[str]:
        # count n-gram frequencies
        ngram_freqs = defaultdict(lambda: 0)
        for entry in entries:
            for ngram in self._extract_n_grams(entry.word + entry.tag):
                ngram_freqs[ngram] += 1
        # select most common n-grams
        ngram_features = \
            list(map(itemgetter(0), 
                     sorted(ngram_freqs.items(), 
                            reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
        return ngram_features

    def _extract_n_grams(self, word :Iterable[str]) -> Iterable[Iterable[str]]:
        result = []
        max_n = min(MAX_NGRAM_LENGTH, len(word))+1
        result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
        result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
        return result

    def save(self, filename :str) -> None:
        # needed for saving: rule IDs
        # (word and edge IDs only needed for preparing the dataset)
        # if present -- save the network and n-gram features
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'NeuralFeatureModel':
        raise NotImplementedError()


class GaussianFeatureModel(FeatureModel):
    def __init__(self, graph :FullGraph) -> None:
        roots = list(graph.nodes_iter())
        edges_by_rule = {}
        for edge in graph.iter_edges():
            if edge.rule not in self.edges_by_rule:
                self.edges_by_rule[edge.rule] = []
            self.edges_by_rule[edge.rule].append(edge)
        # create attribute matrices and indices: roots and edges by rule
        dim = shared.config['Features'].getint('word_vec_dim')
        self.root_idx, self.edge_idx = {}, {}
        self.attr_matrices = {}
        self.attr_matrices['ROOT'] = np.empty((len(roots), dim))
        for i, root in enumerate(roots):
            self.attr_matrices['ROOT'][i,:] = root.vec
            self.root_idx[root] = i
        for rule, edges in edges_by_rule.items():
            self.attr_matrices[rule] = np.empty((len(edges), dim))
            for i, edge in enumerate(edges):
                self.attr_matrices[rule][i,:] = \
                    edge.target.vec - edge.source.vec
                self.edge_idx[edge] = i
        # initial fit
        self.means, self.vars = {}, {}
        for rule in self.attr_matrices:
            self._fit_rule(rule)
        self.recompute_costs()
        self.save_costs_to_file('costs-gaussian.txt')

    def root_cost(self, entry :LexiconEntry) -> float:
        return self.costs['ROOT'][self.root_idx[entry]]

    def edge_cost(self, edge :GraphEdge) -> float:
        return self.costs[edge.rule][self.edge_idx[edge]]

    def recompute_costs(self) -> None:
        self.costs = {}
        for rule, m in self.attr_matrices.items():
            self.costs[rule] = -multivariate_normal.logpdf(\
                                  m, self.means[rule], 
                                  np.diag(self.vars[rule]))

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        weights_by_rule = { rule : np.zeros(m.shape[0]) if rule != 'ROOT' \
                                   else np.ones(m.shape[0]) \
                            for rule, m in self.attr_matrices.items() }
        for edge, weight in sample:
            weights_by_rule[rule][self.edge_idx[edge]] = weight
            weights_by_rule['ROOT'][self.root_idx[edge.target]] -= weight
        for rule, weights in weights_by_rule.items():
            self._fit_rule(rule, weights=weights)

    def save_costs_to_file(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for entry in sorted(self.roots, key=lambda e: e.word):
                write_line(fp, (str(entry), self.root_cost(entry)))
            for rule, edges in self.edges_by_rule.items():
                for edge in edges:
                    edge_cost = self.edge_cost(edge)
                    edge_gain = edge_cost - self.root_cost(edge.target)
                    write_line(fp, (str(edge.source), str(edge.target),
                                    str(edge.rule), edge_cost, edge_gain))

    def _fit_rule(self, rule, weights=None) -> None:
        m = self.matrices[rule]
        if weights is None:
            self.means[rule] = np.sum(m, axis=0) / m.shape[0]
            self.vars[rule] = np.diag(np.dot(m.T, m)) / m.shape[0]
        else:
            sum_weights = np.sum(weights)
            self.means[rule] = np.sum(weights * m.T, axis=1) / sum_weights
            self.vars[rule] = np.diag(np.dot(weights * m.T, m) / sum_weights)

    def save(self, filename :str) -> None:
        # for each rule -- save mean and variance
        # also -- save mean and variance of the ROOT "rule"
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'GaussianFeatureModel':
        raise NotImplementedError()


class ModelSuite:
    def __init__(self, lexicon :Lexicon, edge_set :EdgeSet,
                 rule_set :RuleSet) -> None:
        self.lexicon = lexicon
        self.rule_set = rule_set
        self.root_model = AlergiaRootModel(lexicon)
        self.root_model.fit()
        self.edge_model = BernoulliEdgeModel(edge_set, rule_set)
        self.edge_model.fit_to_sample(np.ones(len(rule_set)))
        self.feature_model = None
        if shared.config['Features'].getfloat('word_vec_weight') > 0:
            self.feature_model = NeuralFeatureModel(edge_set, rule_set)
#             self.feature_model = GaussianFeatureModel(graph)
        self.reset()

    def cost_of_change(self, edges_to_add :List[GraphEdge],
                       edges_to_delete :List[GraphEdge]) -> float:
        result = 0.0
        for edge in edges_to_add:
            result += self.edge_cost(edge) - self.root_cost(edge.target)
        for edge in edges_to_delete:
            result -= self.edge_cost(edge) - self.root_cost(edge.target)
        return result

    def apply_change(self, edges_to_add :List[GraphEdge],
                     edges_to_delete :List[GraphEdge]) -> None:
        self._cost += self.cost_of_change(edges_to_add, edges_to_delete)

    def root_cost(self, entry :LexiconEntry) -> float:
        result = self.root_model.root_cost(entry)
        if self.feature_model is not None:
            result += self.feature_model.root_cost(entry)
        return result

    def rule_cost(self, rule :Rule) -> float:
        return self.edge_model.rule_cost(rule)

    def edge_cost(self, edge :GraphEdge) -> float:
        result = self.edge_model.edge_cost(edge)
        if self.feature_model is not None:
            result += self.feature_model.edge_cost(edge)
        return result

    def recompute_costs(self) -> None:
        self.edge_model.recompute_costs()
        if self.feature_model is not None:
            self.feature_model.recompute_costs()

    def iter_rules(self) -> Iterable[Rule]:
        return iter(self.rule_set)
        
    def cost(self) -> float:
        return self._cost

    def reset(self) -> None:
        self._cost = sum(self.root_cost(entry)\
                        for entry in self.lexicon) +\
                    self.edge_model.null_cost()

#     def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
    def fit_to_sample(self, sample :np.ndarray) -> None:
        self.edge_model.fit_to_sample(sample)
        if self.feature_model is not None:
            self.feature_model.fit_to_sample(sample)

    def fit_to_branching(self, branching :Branching) -> None:
        self.reset()
        self.apply_change(sum(branching.edges_by_rule.values(), []), [])

    def save(self) -> None:
        self.root_model.save(shared.filenames['root-model'])
        self.edge_model.save(shared.filenames['edge-model'])
        if self.feature_model is not None:
            self.feature_model.save(shared.filenames['feature-model'])

    @staticmethod
    def load() -> 'ModelSuite':
        raise NotImplementedError()


# TODO this class is deprecated
# class NeuralModel(Model):
#     def __init__(self, edges :List[GraphEdge]):
#         self.model_type = 'neural'
#         # create rule and edge index
#         self.word_idx = { } # TODO word -> root edge idx
#         self.edge_idx = { edge: idx for idx, edge in enumerate(edges, len(self.word_idx)) }
#         self.rule_idx = { rule: idx for idx, rule in \
#                           enumerate(set(edge.rule for edge in edges)) }
#         self.ngram_features = self.select_ngram_features(edges)
#         print(self.ngram_features)
#         self.features = self.ngram_features
#         self.X_attr, self.X_rule = self.extract_features_from_edges(edges)
#         self.network = self.compile()
#         self.recompute_edge_prob()
# 
#     def fit_to_sample(self, edge_freq :List[Tuple[GraphEdge, float]]) -> None:
#         y = np.empty((len(edge_freq),))
#         for edge, prob in edge_freq:
#             y[self.edge_idx[edge]] = prob
#         self.network.fit([self.X_attr, self.X_rule], y, epochs=5,\
#                          batch_size=100, verbose=1)
# 
#     # TODO rename -> recompute_edge_costs
#     def recompute_edge_prob(self) -> None:
#         self.y_pred = self.network.predict([self.X_attr, self.X_rule])
# 
#     def edge_prob(self, edge :GraphEdge) -> float:
#         return float(self.y_pred[self.edge_idx[edge]])
# 
#     def root_prob(self, node :LexiconEntry) -> float:
#         raise NotImplementedError()
# 
#     def cost() -> float:
#         raise NotImplementedError()
# 
#     def cost_of_change() -> float:
#         raise NotImplementedError()
# 
#     def apply_change() -> None:
#         raise NotImplementedError()
# 
#     def fit_to_branching() -> None:
#         # TODO set the cost to the branching cost
#         raise NotImplementedError()
# 
#     def extract_n_grams(self, word :Iterable[str]) -> Iterable[Iterable[str]]:
#         result = []
#         max_n = min(MAX_NGRAM_LENGTH, len(word))+1
#         result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
#         result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
#         return result
# 
#     def select_ngram_features(self, edges :List[GraphEdge]) -> List[str]:
#         # count n-gram frequencies
#         ngram_freqs = defaultdict(lambda: 0)
#         for edge in edges:
#             source_seq = edge.source.word + edge.source.tag
#             for ngram in self.extract_n_grams(source_seq):
#                 ngram_freqs[ngram] += 1
#         # select most common n-grams
#         ngram_features = \
#             list(map(itemgetter(0), 
#                      sorted(ngram_freqs.items(), 
#                             reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
#         return ngram_features
# 
#     def extract_features_from_edges(self, edges :List[GraphEdge]) -> None:
#         attributes = np.zeros((len(edges), len(self.features)))
#         rule_ids = np.zeros((len(edges), 1))
#         ngram_features_hash = {}
#         for i, ngram in enumerate(self.ngram_features):
#             ngram_features_hash[ngram] = i
#         print('Memory allocation OK.', file=sys.stderr)
#         for i, edge in enumerate(edges):
#             source_seq = edge.source.word + edge.source.tag
#             for ngram in self.extract_n_grams(source_seq):
#                 if ngram in ngram_features_hash:
#                     attributes[i, ngram_features_hash[ngram]] = 1
#             rule_ids[i, 0] = self.rule_idx[edge.rule]
#         print('attributes.nbytes =', attributes.nbytes)
#         print('rule_ids.nbytes =', rule_ids.nbytes)
#         return attributes, rule_ids
# 
#     def sample_negative_examples(self):
#         # shuffle the wordlist
#         # for word in wordlist:
#         #   lookup the word in lexicon .o. rules transducer
#         #   for word2:
#         #     extract all rules from (word, word2)
#         #     for rule:
#         #       if rule in ruleset:
#         #         add (word, rule) to negative examples
#         #         if length(negative examples) >= MAX_NEGATIVE_EXAMPLES:
#         #           prepare the weights vector
#         #           return
#         # if number of examples < MAX_NEGATIVE_EXAMPLES:
#         #   resize the array
#         # prepare the weights vector
#         # return
#         raise NotImplementedError()
# 
#     def compile(self):
#         num_features, num_rules = len(self.features), len(self.rule_idx)
#         input_attr = Input(shape=(num_features,), name='input_attr')
#         dense_attr = Dense(100, activation='softplus', name='dense_attr')(input_attr)
#         input_rule = Input(shape=(1,), name='input_rule')
#         rule_emb = Embedding(input_dim=num_rules, output_dim=100,\
#                              input_length=1)(input_rule)
#         rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
#         concat = keras.layers.concatenate([dense_attr, rule_emb_fl])
#         dense = Dense(100, activation='softplus', name='dense')(concat)
#         output = Dense(1, activation='sigmoid', name='output')(dense)
# 
#         model = Model(inputs=[input_attr, input_rule], outputs=[output])
#         model.compile(optimizer='adam', loss='binary_crossentropy')
#         return model


