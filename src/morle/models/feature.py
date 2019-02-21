from morle.datastruct.graph import EdgeSet, FullGraph, GraphEdge
from morle.datastruct.lexicon import Lexicon, LexiconEntry
from morle.datastruct.rules import RuleSet, Rule
from morle.models.generic import Model, ModelFactory, UnknownModelTypeException
from morle.utils.files import full_path
import morle.shared as shared

import keras.models
from keras.layers import concatenate, Dense, Embedding, Flatten, Input, \
                         SimpleRNN
import keras
import logging
import numpy as np
import os.path
from scipy.stats import multivariate_normal
from typing import Any, Dict, Iterable, List, Tuple


class RootFeatureModel(Model):
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
        y = self._prepare_data(lexicon)
        self.mean = np.average(y, weights=weights, axis=0)
        err = y - self.mean
        self.var = np.average(err**2, weights=weights, axis=0)

    def root_cost(self, entry :LexiconEntry) -> float:
        return -multivariate_normal.logpdf(entry.vec, self.mean,
                                           np.diag(self.var))

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        y = self._prepare_data(lexicon)
        return -multivariate_normal.logpdf(y, self.mean, np.diag(self.var))

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

    def _prepare_data(self, lexicon :Lexicon) -> np.ndarray:
        return np.vstack([entry.vec for entry in lexicon])


class RNNRootFeatureModel(RootFeatureModel):
    # TODO alphabet_hash: include the unknown symbol?
    def __init__(self, alphabet :Iterable[str] = None, maxlen :int = None) \
                -> None:
        self.alphabet = alphabet
        self.alphabet_hash = dict((y, x) for (x, y) in enumerate(alphabet, 1))
        self.dim = shared.config['Features'].getint('word_vec_dim')
        self.maxlen = maxlen                        
        self._compile_network()

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        # prepare data
        X, y = self._prepare_data(lexicon)
        # fit the neural network
        self.nn.fit(X, y, epochs=5, sample_weight=weights, batch_size=64,
                    verbose=1)
        # fit error variance
        y_pred = self.nn.predict(X)
        err = y-y_pred
        self.err_var = np.average(err**2, axis=0, weights=weights)

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        X, y = self._prepare_data(lexicon)
        y_pred = self.nn.predict(X)
        return -multivariate_normal.logpdf(y-y_pred,
                                           np.zeros(self.dim),
                                           np.diag(self.err_var))

    def save(self, filename) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        weights = self.nn.get_weights()
        np.savez(file_full_path, emb=weights[0], rnn_1=weights[1],
                 rnn_2=weights[2], rnn_3=weights[3], d1_1=weights[4],
                 d1_2=weights[5], d2=weights[6], err_var=self.err_var)
        
    @staticmethod
    def load(filename :str,
             alphabet :Iterable[str] = None,
             maxlen :int = None) -> 'RNNRootFeatureModel':
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        data = np.load(file_full_path)
        result = RNNRootFeatureModel(alphabet, maxlen)
        result.nn.layers[0].set_weights([data['emb']])
        result.nn.layers[1].set_weights([data['rnn_1'], data['rnn_2'],
                                         data['rnn_3']])
        result.nn.layers[2].set_weights([data['d1_1'], data['d1_2']])
        result.nn.layers[3].set_weights([data['d2']])
        result.err_var = data['err_var']
        return result

    def _compile_network(self) -> None:
        self.nn = keras.models.Sequential()
        self.nn.add(Embedding(input_dim=len(self.alphabet)+1,
                              output_dim=self.dim, mask_zero=True,
                              input_length=self.maxlen))
        self.nn.add(SimpleRNN(self.dim, activation='relu',
                              return_sequences=False))
        self.nn.add(Dense(self.dim, use_bias=True, activation='tanh'))
        self.nn.add(Dense(self.dim, use_bias=False, activation='linear'))
        self.nn.compile(loss='mse', optimizer='adam')

    def _prepare_data(self, lexicon :Lexicon) -> Tuple[np.ndarray, np.ndarray]:
        X_lst, y_lst = [], []
        for entry in lexicon:
            X_lst.append([self.alphabet_hash[sym] \
                          for sym in entry.word+entry.tag])
            y_lst.append(entry.vec)
        X = keras.preprocessing.sequence.pad_sequences(X_lst)
        y = np.vstack(y_lst)
        return X, y


class RootFeatureModelFactory(ModelFactory):
    @staticmethod
    def get_rnn_model_parameters(lexicon :Lexicon) -> Dict[str, Any]:
        result = {}
        if lexicon is not None:
            result['alphabet'] = lexicon.get_alphabet()
            logging.getLogger('main').debug(\
                'Detected alphabet: {}'.format(', '.join(result['alphabet'])))
            result['maxlen'] = lexicon.get_max_symstr_length()
            logging.getLogger('main').debug(\
                'Detected max. word length: {}'.format(result['maxlen']))
        else:
            # default settings (TODO move somewhere else?)
            result['alphabet'] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                                  'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                  's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] + \
                                 shared.multichar_symbols
            result['maxlen'] = 20
        return result

    @staticmethod
    def create(model_type :str, lexicon :Lexicon = None) -> RootFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'gaussian':
            return GaussianRootFeatureModel()
        elif model_type == 'rnn':
            params = RootFeatureModelFactory.get_rnn_model_parameters(lexicon)
            return RNNRootFeatureModel(**params)
        else:
            raise UnknownModelTypeException('root feature', model_type)

    @staticmethod
    def load(model_type :str, filename :str, lexicon :Lexicon = None) \
            -> RootFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'gaussian':
            return GaussianRootFeatureModel.load(filename)
        elif model_type == 'rnn':
            params = RootFeatureModelFactory.get_rnn_model_parameters(lexicon)
            return RNNRootFeatureModel.load(filename, **params)
        else:
            raise UnknownModelTypeException('root feature', model_type)


class EdgeFeatureModel(Model):
    def predict_target_feature_vec(self, edge :GraphEdge) -> np.ndarray:
        raise NotImplementedError()


class NeuralEdgeFeatureModel(EdgeFeatureModel):
    def __init__(self, rule_set :RuleSet) -> None:
        self.rule_set = rule_set
        self.err_var = None
        self._compile_network()

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        X_attr, X_rule, y = self._prepare_data(edge_set)
        # fit the predictor
        self.nn.fit([X_attr, X_rule], y, epochs=20, sample_weight=weights,
                     batch_size=1000, verbose=0)
        # fit the error
        y_pred = self.nn.predict([X_attr, X_rule])
        err = y-y_pred
        self.err_var = np.average(err**2, axis=0, weights=weights)

    # TODO unused?
    def edge_cost(self, edge :GraphEdge) -> float:
        X_attr = np.array([edge.source.vec])
        X_rule = np.array([self.rule_set.get_id(edge.rule)])
        y = np.array([edge.target.vec])
        y_pred = self.nn.predict([X_attr, X_rule])
        return -multivariate_normal.logpdf(y-y_pred, np.zeros(y.shape[1]),
                                           np.diag(self.err_var))

    def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
        X_attr, X_rule, y = self._prepare_data(edge_set)
        y_pred = self.nn.predict([X_attr, X_rule])
        return -multivariate_normal.logpdf(y-y_pred, np.zeros(y.shape[1]),
                                           np.diag(self.err_var))

    def predict_target_feature_vec(self, edge :GraphEdge) -> np.ndarray:
        # TODO remove code duplication with edge_cost()
        X_attr = np.array([edge.source.vec])
        X_rule = np.array([self.rule_set.get_id(edge.rule)])
        y_pred = self.nn.predict([X_attr, X_rule])
        return y_pred

    def save(self, filename) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        weights = self.nn.get_weights()
        np.savez(file_full_path, rule_emb=weights[0], d_1=weights[1],
                 d_2=weights[2], err_var=self.err_var)
    
    @staticmethod
    def load(filename, rule_set :RuleSet) -> 'NeuralEdgeFeatureModel':
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        data = np.load(file_full_path)
        result = NeuralEdgeFeatureModel(rule_set)
        result.nn.layers[1].set_weights([data['rule_emb']])
        result.nn.layers[5].set_weights([data['d_1'], data['d_2']])
        result.err_var = data['err_var']
        return result

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

        self.nn = keras.models.Model(inputs=[input_attr, input_rule], outputs=[output])
        self.nn.compile(optimizer='adam', loss='mse')

    def _prepare_data(self, edge_set :EdgeSet) -> \
                     Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # build the arrays as lists
        X_attr_lst, X_rule_lst, y_lst = [], [], []
        for edge in edge_set:
            X_attr_lst.append(edge.source.vec)
            X_rule_lst.append(self.rule_set.get_id(edge.rule))
            y_lst.append(edge.target.vec)
        # convert the lists into matrices
        X_attr = np.vstack(X_attr_lst)
        X_rule = np.array(X_rule_lst)
        y = np.vstack(y_lst)
        return X_attr, X_rule, y


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
            feature_matrix = np.array([edge_set[i].target.vec - \
                                       edge_set[i].source.vec \
                                       for i in edge_ids])
            self.fit_rule(self.rule_set.get_id(rule), feature_matrix,
                          weights[edge_ids,])

    def edge_cost(self, edge :GraphEdge) -> float:
        rule_id = self.rule_set.get_id(edge.rule)
        return -multivariate_normal.logpdf(edge.attr['vec'],
                                           self.means[rule_id,],
                                           np.diag(self.vars[rule_id,]))

    def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
        result = np.zeros(len(edge_set))
        for rule, edge_ids in edge_set.get_edge_ids_by_rule().items():
            rule_id = self.rule_set.get_id(rule)
            feature_matrix = np.vstack([edge_set[i].target.vec - \
                                        edge_set[i].source.vec \
                                        for i in edge_ids])
            costs = -multivariate_normal.logpdf(feature_matrix,
                                                self.means[rule_id,],
                                                np.diag(self.vars[rule_id,]))
            result[tuple(edge_ids),] = costs
        return result

    def predict_target_feature_vec(self, edge :GraphEdge) -> np.ndarray:
        rule_id = self.rule_set.get_id(edge.rule)
        return edge.source.vec + self.means[rule_id,]

    def save(self, filename) -> None:
        np.savez(full_path(filename), means=self.means, vars=self.vars)

    @staticmethod
    def load(filename, rule_set :RuleSet) -> 'GaussianEdgeFeatureModel':
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        result = GaussianEdgeFeatureModel(rule_set)
        with np.load(file_full_path) as data:
            result.means = data['means']
            result.vars = data['vars']
        return result


class EdgeFeatureModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str, rule_set :RuleSet) -> EdgeFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'gaussian':
            return GaussianEdgeFeatureModel(rule_set)
        elif model_type == 'neural':
            return NeuralEdgeFeatureModel(rule_set)
        else:
            raise UnknownModelTypeException('edge feature', model_type)

    @staticmethod
    def load(model_type :str, filename :str, rule_set :RuleSet) \
            -> EdgeFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'gaussian':
            return GaussianEdgeFeatureModel.load(filename, rule_set)
        elif model_type == 'neural':
            return NeuralEdgeFeatureModel.load(filename, rule_set)
        else:
            raise UnknownModelTypeException('edge feature', model_type)

