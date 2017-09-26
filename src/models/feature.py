from datastruct.graph import EdgeSet, FullGraph, GraphEdge
from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.rules import RuleSet, Rule
import shared

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
        with np.load(file_full_path) as data:
            self.mean = data['mean']
            self.var = data['var']


class RNNRootFeatureModel(RootFeatureModel):
    # TODO a character-level RNN for predicting vector features
    def __init__(self) -> None:
        raise NotImplementedError()

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        raise NotImplementedError()


class EdgeFeatureModel:
    pass


class NeuralEdgeFeatureModel(EdgeFeatureModel):
    def __init__(self) -> None:
        raise NotImplementedError()

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        raise NotImplementedError()


class GaussianEdgeFeatureModel(EdgeFeatureModel):
    def __init__(self) -> None:
        self.means = {}
        self.vars = {}

    def fit_rule(self, rule :Rule, feature_matrix :np.ndarray,
                 weights :np.ndarray) -> None:
        if np.sum(weights > 0) <= 1:
            logging.getLogger('main').debug(
                'GaussianEdgeFeatureModel: rule {} cannot be fitted:'
                ' not enough edges.'.format(rule))
            return;
        self.means[rule] = np.average(feature_matrix, weights=weights, axis=0)
        err = feature_matrix - self.means[rule]
        self.vars[rule] = np.average(err**2, weights=weights, axis=0) +\
                          np.ones(err.shape[1])

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        for rule, edge_ids in edge_set.get_edge_ids_by_rule().items():
            edge_ids = tuple(edge_ids)
            self.fit_rule(rule, edge_set.feature_matrix[edge_ids,:], 
                          weights[edge_ids,])

    def edge_cost(self, edge :GraphEdge) -> float:
        return -multivariate_normal.logpdf(edge.attr['vec'],
                                           self.means[edge.rule],
                                           np.diag(self.vars[edge.rule]))

    def save(self, filename):
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        np.savez(file_full_path, means=self.means, vars=self.vars)

    @staticmethod
    def load(filename):
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        with np.load(file_full_path) as data:
            self.means = data['means']
            self.varS = data['vars']


class FeatureModel:
    def __init__(self, graph :FullGraph) -> None:
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def recompute_costs(self) -> None:
        raise NotImplementedError()

    def fit_to_sample(self, root_weights :np.ndarray, 
                      edge_weights :np.ndarray) -> None:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'FeatureModel':
        raise NotImplementedError()


# TODO also: NeuralFeatureModel with an individual per-rule variance?
class NeuralFeatureModel(FeatureModel):
    def __init__(self, rule_set :RuleSet) -> None:
#         self.lexicon = lexicon
#         self.edge_set = edge_set
        self.rule_set = rule_set
        self._prepare_data(lexicon, edge_set, rule_set)

#     def root_cost(self, entry :LexiconEntry) -> float:
#         return float(self.costs[self.lexicon.get_id(entry)])
# 
#     def edge_cost(self, edge :GraphEdge) -> float:
#         return float(self.costs[len(self.lexicon) +\
#                                 self.edge_set.get_id(edge)])

#     def recompute_costs(self) -> None:
#         self.y_pred = self.model.predict([self.X_attr, self.X_rule])
#         self._fit_error()
#         error = self.y - self.y_pred
#         self.costs = \
#             -multivariate_normal.logpdf(error, np.zeros(self.y.shape[1]),
#                                         self.error_cov)

    def fit(self, root_weights :np.ndarray, 
                      edge_weights :np.ndarray) -> None:
        weights = np.hstack((root_weights, edge_weights))
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

    def _prepare_data(self, lexicon :Lexicon, edge_set :EdgeSet,
                      rule_set :RuleSet) -> None:
        self.ngram_features = self._select_ngram_features(graph.nodes_iter())
        ngram_features_hash = {}
        for i, ngram in enumerate(self.ngram_features):
            ngram_features_hash[ngram] = i
        vector_dim = shared.config['Features'].getint('word_vec_dim')
        sample_size = len(self.word_set) + len(self.edge_set)
        num_features = len(self.ngram_features) +\
                       shared.config['Features'].getint('word_vec_dim')
        self.X_attr = np.zeros((sample_size, num_features))
        self.X_rule = np.empty(sample_size)
        self.y = np.empty((sample_size, vector_dim))
        for idx, entry in enumerate(self.lexicon):
            for ngram in self._extract_n_grams(entry.word):
                if ngram in ngram_features_hash:
                    self.X_attr[idx, ngram_features_hash[ngram]] = 1
            self.X_rule[idx] = 0
            self.y[idx] = entry.vec
        for idx, edge in enumerate(self.edge_set, len(self.lexicon)):
            for ngram in self._extract_n_grams(edge.source.word):
                if ngram in ngram_features_hash:
                    self.X_attr[idx, ngram_features_hash[ngram]] = 1
            self.X_attr[idx, len(ngram_features_hash):] = edge.source.vec
            self.X_rule[idx] = self.rule_set.get_id(edge.rule) + 1
            self.y[idx] = edge.target.vec

    def compile(self) -> None:
        vector_dim = shared.config['Features'].getint('word_vec_dim')
        num_rules = len(self.rule_set)+1
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

    def _fit_error(self) -> None:
        n = self.y.shape[0]
        error = self.y - self.y_pred
        self.error_cov = np.dot(error.T, error)/n

    def _select_ngram_features(self, entries :Iterable[LexiconEntry]) \
                              -> List[str]:
        # count n-gram frequencies
        ngram_freqs = defaultdict(lambda: 0)
        for entry in entries:
            for ngram in self._extract_n_grams(entry.word):
                ngram_freqs[ngram] += 1
        # select most common n-grams
        ngram_features = \
            list(map(itemgetter(0), 
                     sorted(ngram_freqs.items(), 
                            reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
        return ngram_features

    def _extract_n_grams(self, word :List[str]) -> List[List[str]]:
        result = []
        max_n = min(MAX_NGRAM_LENGTH, len(word))+1
        result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
        result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
        return result

    def save(self, filename :str) -> None:
        self.model.save(filename)
        # TODO n-gram features
        raise NotImplementedError()
        self.save_ngram_features(filename + '.ngrams')

    def save_ngram_features(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for ngram in self.ngram_features:
                write_line(fp, (ngram,))

    @staticmethod
    def load(filename :str, lexicon :Lexicon, edge_set :EdgeSet,
             rule_set :RuleSet) -> 'NeuralFeatureModel':
        result = NeuralFeatureModel(lexicon, edge_set, rule_set)
        result.model = keras.models.load_model(filename)
        result.ngrams = load_ngram_features(filename + '.ngrams')
        # TODO ngram_features_hash
        # TODO isolate all n-gram-related things to a new class
        # (NGramFeatureSet or sth)
        raise NotImplementedError()

    @staticmethod
    def load_ngram_features(filename :str) -> List[List[str]]:
        result = []
        for ngram in read_tsv_file(filename):
            result.append(ngram)
        return result

# TODO refactor according to the new indexing

class GaussianFeatureModel(FeatureModel):
    def __init__(self, lexicon :Lexicon, edge_set :EdgeSet,
                 rule_set :RuleSet) -> None:
        self.lexicon = lexicon
        self.edge_set = edge_set
        self.rule_set = rule_set
        # group edge ids by rule
        self.edge_ids_by_rule = [[] for i in range(len(rule_set))]
        for edge_id, edge in enumerate(edge_set):
            rule_id = self.rule_set.get_id(edge.rule)
            self.edge_ids_by_rule[rule_id].append(edge_id)
        # create a feature value matrix for roots and for each rule
        dim = shared.config['Features'].getint('word_vec_dim')
        self.attr = []
        root_attr_matrix = np.empty((len(lexicon), dim))
        for i, entry in enumerate(lexicon):
            root_attr_matrix[i,] = entry.vec
        self.attr.append(root_attr_matrix)
        for edges in self.edge_ids_by_rule:
            attr_matrix = np.empty((len(edges), dim))
            for i, edge_id in enumerate(edges):
                edge = self.edge_set[edge_id]
                attr_matrix[i,] = edge.target.vec - edge.source.vec
            self.attr.append(attr_matrix)
        # initial parameter values
        self.means = np.zeros((len(rule_set)+1, dim))
        self.vars = np.ones((len(rule_set)+1, dim))

    def root_cost(self, entry :LexiconEntry) -> float:
        return self.costs[self.lexicon.get_id(entry)]

    def edge_cost(self, edge :GraphEdge) -> float:
        return self.costs[len(self.lexicon) + self.edge_set.get_id(edge)]

    # TODO needs optimization?
    def recompute_costs(self) -> None:
        logging.getLogger('main').info('Recomputing costs...')
        self.costs = np.empty(len(self.lexicon) + len(self.edge_set))
        for idx, entry in enumerate(self.lexicon):
            self.costs[idx,] = -multivariate_normal.logpdf(\
                                  self.attr[0][idx,], self.means[0], 
                                  np.diag(self.vars[0]))
        progressbar = tqdm.tqdm(total=len(self.edge_ids_by_rule))
        for rule_id, edge_ids in enumerate(self.edge_ids_by_rule, 1):
            for idx, edge_id in enumerate(edge_ids):
                self.costs[len(self.lexicon)+edge_id,] =\
                    -multivariate_normal.logpdf(self.attr[rule_id][idx,],
                                                self.means[rule_id],
                                                np.diag(self.vars[rule_id]))
            progressbar.update()
        progressbar.close()

    def initial_fit(self):
        self.fit_to_sample(np.ones(len(self.lexicon)),
                           np.ones(len(self.edge_set)))

    def fit_to_sample(self, root_weights :np.ndarray, 
                      edge_weights :np.ndarray) -> None:
        weights_by_rule = [root_weights] +\
                          [np.zeros(len(edge_ids)) \
                             for edge_ids in self.edge_ids_by_rule]
        # group edge weights by rule
        for rule_id, edge_ids in enumerate(self.edge_ids_by_rule):
            for idx, edge_id in enumerate(edge_ids):
                rule_id = self.rule_set.get_id(self.edge_set[edge_id].rule)
                weights_by_rule[rule_id+1][idx] = edge_weights[edge_id]
        for rule_id, weights in enumerate(weights_by_rule):
            if np.sum(weights > 0) > 1:
                self.means[rule_id] = np.average(self.attr[rule_id],
                                                 weights=weights, axis=0)
                self.vars[rule_id] = \
                    np.average((self.attr[rule_id]-self.means[rule_id])**2,
                               weights=weights, axis=0) + 1 #TODO a proper prior!!!
        self.recompute_costs()

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

    def save(self, filename :str) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        np.savez(file_full_path, means=self.means, vars=self.vars)

    def load(self, filename :str) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        with np.load(file_full_path) as data:
            self.means = data['means']
            self.vars = data['vars']

