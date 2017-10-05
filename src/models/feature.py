from datastruct.graph import EdgeSet, FullGraph, GraphEdge
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import RuleSet, Rule
import shared

from keras.models import Model, Sequential
from keras.layers import concatenate, Dense, Embedding, Flatten, Input, \
                         SimpleRNN
import keras
import logging
import numpy as np
import os.path
from scipy.stats import multivariate_normal
from typing import Iterable, List


class RootFeatureModel:
    pass


class NeuralRootFeatureModel(RootFeatureModel):
    def __init__(self) -> None:
        pass

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        raise NotImplementedError()


class GaussianRootFeatureModel(RootFeatureModel):
    def __init__(self) -> None:
        dim = shared.config['Features'].getint('word_vec_dim')
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        self.mean = np.average(lexicon.feature_matrix, weights=weights, axis=0)
        err = lexicon.feature_matrix - self.mean
        self.var = np.average(err**2, weights=weights, axis=0)

    def root_cost(self, entry :LexiconEntry) -> float:
        return -multivariate_normal.logpdf(entry.vec, self.mean,
                                           np.diag(self.var))

    def save(self, filename):
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        np.savez(file_full_path, mean=self.mean, var=self.var)

    @staticmethod
    def load(filename):
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        result = GaussianRootFeatureModel()
        with np.load(file_full_path) as data:
            result.mean = data['mean']
            result.var = data['var']
        return result


class RNNRootFeatureModel(RootFeatureModel):
    def __init__(self, alphabet :Iterable[str], maxlen :int) -> None:
        self.alphabet = alphabet
        self.alphabet_hash = dict((y, x) for (x, y) in enumerate(alphabet, 1))
        self.dim = shared.config['Features'].getint('word_vec_dim')
        self.maxlen = maxlen                        
        self._compile_network()

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        # prepare data
        X, y = [], []
        for entry in lexicon:
            X.append([self.alphabet_hash[sym] for sym in entry.word+entry.tag])
            y.append(entry.vec)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=self.maxlen)
        y = np.vstack(y)
        # fit the neural network
        self.nn.fit(X, y, epochs=20, sample_weight=weights, batch_size=64,
                    verbose=1)
        # fit error variance
        y_pred = self.nn.predict(X)
        err = y-y_pred
        self.err_var = np.average(err**2, axis=0, weights=weights)

    def root_costs(self, entries :Iterable[LexiconEntry]) -> np.ndarray:
        X, y = [], []
        for entry in entries:
            X.append([self.alphabet_hash[sym] for sym in entry.word+entry.tag])
            y.append(entry.vec)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=self.maxlen)
        y = np.vstack(y)
        y_pred = self.nn.predict(X)
        return -multivariate_normal.logpdf(y-y_pred,
                                           np.zeros(self.dim),
                                           np.diag(self.err_var))

#     def root_cost(self, entry :LexiconEntry) -> float:
#         X = [[self.alphabet_hash[sym] for sym in entry.word+entry.tag]]
#         X = keras.preprocessing.sequence.pad_sequences(X, maxlen=self.maxlen)
#         y_pred = self.nn.predict(X)
#         return -multivariate_normal.logpdf(entry.vec-y_pred,
#                                            np.zeros(self.dim),
#                                            np.diag(self.err_var))

    def save(self, filename) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        iw = self.nn.get_weights()
        np.savez(file_full_path, emb=iw[0], internal=iw[1], output=iw[2],
                 err_var=self.err_var)
        
    @staticmethod
    def load(filename) -> 'RNNRootFeatureModel':
        raise NotImplementedError()

    def _compile_network(self) -> None:
        self.nn = Sequential()
        self.nn.add(Embedding(input_dim=len(self.alphabet)+1,
                              output_dim=self.dim, mask_zero=True,
                              input_length=self.maxlen))
        self.nn.add(SimpleRNN(self.dim, activation='relu',
                              return_sequences=False))
        self.nn.add(Dense(self.dim, use_bias=True, activation='tanh'))
        self.nn.add(Dense(self.dim, use_bias=False, activation='linear'))
        self.nn.compile(loss='mse', optimizer='adam')


class EdgeFeatureModel:
    pass


class NeuralEdgeFeatureModel(EdgeFeatureModel):
    def __init__(self, rule_set :RuleSet) -> None:
        self.rule_set = rule_set
        self.err_var = None
        self._compile_network()

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        # convert input and output data to matrices
        X_attr, X_rule, y = [], [], []
        for edge in edge_set:
            X_attr.append(edge.source.vec)
            X_rule.append(self.rule_set.get_id(edge.rule))
            y.append(edge.target.vec)
        X_attr = np.vstack(X_attr)
        X_rule = np.array(X_rule)
        y = np.vstack(y)
        # fit the predictor
        self.nn.fit([X_attr, X_rule], y, epochs=20, sample_weight=weights,
                     batch_size=1000, verbose=0)
        # fit the error
        y_pred = self.nn.predict([X_attr, X_rule])
        err = y-y_pred
        self.err_var = np.average(err**2, axis=0, weights=weights)

    # TODO compute costs for a whole EdgeSet/Lexicon!

    def edge_cost(self, edge :GraphEdge) -> float:
        X_attr = np.array([edge.source.vec])
        X_rule = np.array([self.rule_set.get_id(edge.rule)])
        y = np.array([edge.target.vec])
        y_pred = self.nn.predict([X_attr, X_rule])
        return -multivariate_normal.logpdf(y-y_pred, np.zeros(y.shape[1]),
                                           np.diag(self.err_var))

    def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
        X_attr, X_rule, y = [], [], []
        for edge in edge_set:
            X_attr.append(edge.source.vec)
            X_rule.append(self.rule_set.get_id(edge.rule))
            y.append(edge.target.vec)
        X_attr = np.vstack(X_attr)
        X_rule = np.array(X_rule)
        y = np.vstack(y)
        y_pred = self.nn.predict([X_attr, X_rule])
        return -multivariate_normal.logpdf(y-y_pred, np.zeros(y.shape[1]),
                                           np.diag(self.err_var))

    def save(self, filename) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        iw = self.nn.get_weights()
        np.savez(file_full_path, rule_emb=iw[0], internal=iw[1], output=iw[2],
                 err_var=self.err_var)
    
    @staticmethod
    def load(filename, rule_set :RuleSet) -> 'NeuralEdgeFeatureModel':
        raise NotImplementedError()

    def _compile_network(self):
        dim = shared.config['Features'].getint('word_vec_dim')
        num_rules = len(self.rule_set)
        input_attr = Input(shape=(dim,), name='input_attr')
        input_rule = Input(shape=(1,), name='input_rule')
        rule_emb = Embedding(input_dim=num_rules, output_dim=100,\
                             input_length=1)(input_rule)
        rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
        concat = concatenate([input_attr, rule_emb_fl])
        output = Dense(dim, activation='linear', name='dense')(concat)

        self.nn = Model(inputs=[input_attr, input_rule], outputs=[output])
        self.nn.compile(optimizer='adam', loss='mse')


class GaussianEdgeFeatureModel(EdgeFeatureModel):
    def __init__(self, rule_set :RuleSet) -> None:
        self.dim = shared.config['Features'].getint('word_vec_dim')
        self.rule_set = rule_set
        self.means = None
        self.vars = None

    def fit_rule(self, rule_id :int, feature_matrix :np.ndarray,
                 weights :np.ndarray) -> None:
        if np.sum(weights > 0) <= 1:
            logging.getLogger('main').debug(
                'GaussianEdgeFeatureModel: rule {} cannot be fitted:'
                ' not enough edges.'.format(self.rule_set[rule_id]))
            return
        self.means[rule_id,] = np.average(feature_matrix, weights=weights,
                                          axis=0)
        err = feature_matrix - self.means[rule_id,]
        self.vars[rule_id,] = np.average(err**2, weights=weights, axis=0) +\
                              0.001 * np.ones(self.dim)

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) \
           -> None:
        if self.means is None:
            self.means = np.empty((len(self.rule_set), self.dim))
        if self.vars is None:
            self.vars = np.empty((len(self.rule_set), self.dim))
        for rule, edge_ids in edge_set.get_edge_ids_by_rule().items():
            edge_ids = tuple(edge_ids)
            feature_matrix = edge_set.feature_matrix[edge_ids,:]
            self.fit_rule(self.rule_set.get_id(rule),
                          edge_set.feature_matrix[edge_ids,:], 
                          weights[edge_ids,])

    def edge_cost(self, edge :GraphEdge) -> float:
        rule_id = self.rule_set.get_id(edge.rule)
        return -multivariate_normal.logpdf(edge.attr['vec'],
                                           self.means[rule_id,],
                                           np.diag(self.vars[rule_id,]))

    def save(self, filename) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        np.savez(file_full_path, means=self.means, vars=self.vars)

    @staticmethod
    def load(filename, rule_set :RuleSet) -> 'GaussianEdgeFeatureModel':
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        result = GaussianEdgeFeatureModel()
        with np.load(file_full_path) as data:
            result.means = data['means']
            result.vars = data['vars']
        return result

