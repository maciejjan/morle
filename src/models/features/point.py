from models.features.generic import *
import shared

import numpy as np
from scipy.special import betaln, gammaln
from typing import Any


class PointFeature(Feature):
    def cost(self, values :Any) -> float:
        raise NotImplementedError()

    def cost_of_change(self, values_to_add :Any, values_to_remove :Any) \
                      -> float:
        return self.cost(values_to_add) - self.cost(values_to_remove)

    def apply_change(self, values_to_add :Any, values_to_remove :Any) -> None:
        pass

    def reset(self) -> None:
        pass


class PointBinomialFeature(PointFeature):

    def __init__(self, trials :int, prob :float = 0.5, 
                 alpha :float = 1, beta :float = 1) -> None:
        self.alpha = alpha
        self.beta = beta
        self.trials = trials
        self.prob = prob
    
    def cost(self, value :float) -> float:
        # TODO  int or list?
#         value = sum(values)
        return -value*(np.log(self.prob)-np.log(1-self.prob))

    def empty(self) -> int:
        return 0
    
    def weighted_cost(self, value :float) -> float:
        # TODO  int or list?
#         value = sum(val*w for val, w in values)
        return -value*(np.log(self.prob)-np.log(1-self.prob))
    
    # the normalizing constant of the distribution
    def null_cost(self) -> float:
        return betaln(self.alpha, self.beta)-\
            (self.alpha-1)*np.log(self.prob) - \
            (self.trials+self.beta-1) * np.log(1-self.prob)
    
    def update(self, prob :float) -> None:
        self.prob = prob
    
    def fit(self, value :float) -> None:
        self.prob = (value + self.alpha-1) / (self.trials + self.alpha + self.beta - 2)
#        self.prob = min(max(self.prob, 1e-10), 0.9999)
    
    def weighted_fit(self, value :float) -> None:
        self.fit(value)
#         self.prob = (sum(val*w for val, w in values) + self.alpha - 1) / (self.trials + self.alpha + self.beta - 2)
#        self.prob = min(max(self.prob, 1e-10), 0.9999)
    
#     def num_args(self):
#         return 2
#     
#     def parse_string_args(self, trials, prob):
#         self.trials = int(trials)
#         self.prob = float(prob)

    def to_string(self):
        return '\t'.join((str(self.trials), str(self.prob)))


# TODO refactor
class PointExponentialFeature(PointFeature):
    def __init__(self, rate=1.0):
        self.rate = rate

    def cost(self, values):
        return -sum(np.log(self.rate) + self.rate*val for val in values)
    
    # TODO add/remove edges

    def fit(self, values):
        if not values: return
        values = np.asarray(list(values))
        self.rate = 1.0 / np.mean(values)


# TODO refactor
class PointGaussianInverseChiSquaredFeature(PointFeature):
    '''Multivariate Gaussian feature with independent coordinates,
       zero-centered prior on means and Inv-Chi-Sq prior on variances.'''

    def __init__(self, dim, kappa_0, mu_0, nu_0, var_0):
        self.dim = dim
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.var_0 = var_0
        self.mean = kappa_0 * mu_0 * np.ones(self.dim)
        self.var = nu_0 * var_0 * np.ones(self.dim)
    
    def statistics(self, values):
        matrix = np.stack(values)
        shape = (matrix.shape[0],) if self.dim == 1 else (matrix.shape[0], 1)
        s_0 = matrix.shape[0]
        s_1 = np.sum(matrix, axis=0) / s_0
        s_2 = np.sum((matrix-np.tile(s_1, shape))**2, axis=0) / s_0
        if self.dim == 1:
            s_1, s_2 = s_1 * np.ones(self.dim), s_2 * np.ones(self.dim)
        return s_0, s_1, s_2
    
    def weighted_statistics(self, values):
        matrix = np.stack([val for val, w in values])
        weights = np.array([w for val, w in values])
        shape = (matrix.shape[0],) if self.dim == 1 else (matrix.shape[0], 1)
        s_0 = np.sum(weights)
        if s_0 == 0:
            return 0, None, None
        s_1 = np.average(matrix, axis=0, weights=weights)
        s_2 = np.average((matrix-np.tile(s_1, shape))**2, axis=0, weights=weights)
        if self.dim == 1:
            s_1, s_2 = s_1 * np.ones(self.dim), s_2 * np.ones(self.dim)
        return s_0, s_1, s_2
    
    def fit_with_statistics(self, s_0, s_1, s_2):
        self.mean = (self.kappa_0*self.mu_0*np.ones(self.dim) +\
            s_0*s_1) / (self.kappa_0 + s_0)
        self.var = (self.nu_0*self.var_0*np.ones(self.dim) +\
            s_0*s_2 + s_0*self.kappa_0/(s_0+self.kappa_0)*\
                (self.mu_0*np.ones(self.dim)-s_1)**2) /\
            (self.nu_0 + s_0 - 1)
    
    ### coordinate-wise functions ###

    def coordinate_null_cost(self, mean, var):
        return -0.5*np.log(0.5*self.kappa_0/np.pi) + gammaln(0.5*self.nu_0) -\
            0.5*self.nu_0*np.log(0.5*self.nu_0*self.var_0) +\
            (0.5*self.nu_0+1.5)*np.log(var) +\
            0.5/var*(self.nu_0*self.var_0 +\
                self.kappa_0*(self.mu_0-mean)**2)
    
    def coordinate_cost(self, mean, var, s_0, s_1, s_2):
        return 0.5*s_0*np.log(2*np.pi*var) +\
             0.5*s_0/var*(s_2 + (s_1 - mean)**2)

    ### main methods (for export) ###

    def null_cost(self):
        return sum(self.coordinate_null_cost(self.mean[i], self.var[i])
                   for i in range(self.dim))

    def cost(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        if s_0 == 0:
            return 0
        return sum(self.coordinate_cost(\
                                        self.mean[i], self.var[i],
                                        s_0, s_1[i], s_2[i])\
                   for i in range(self.dim))

    def weighted_cost(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        if s_0 == 0:
            return 0
        return sum(self.coordinate_cost(\
                                        self.mean[i], self.var[i],
                                        s_0, s_1[i], s_2[i])\
                   for i in range(self.dim))

    def fit(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        if s_0 > 0:
            self.fit_with_statistics(s_0, s_1, s_2)

    def weighted_fit(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        if s_0 > 0:
            self.fit_with_statistics(s_0, s_1, s_2)

    def num_args(self):
        raise Exception('Not implemented!')

    def to_string(self):
        return '\t'.join((
            ' '.join(map(str, self.mean)),
            ' '.join(map(str, self.var))))


class PointFeatureSet(FeatureSet):
    '''Creates sets of features according to the program configuration.'''

    def __init__(self):
        self.features = []
        self.weights = []
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    @staticmethod
    def new_edge_feature_set(domsize :int) -> 'PointFeatureSet':
        result = PointFeatureSet()
        features = [PointBinomialFeature(domsize, alpha=1.1, beta=1.1)]
        weights = [1.0]
        if shared.config['Features'].getfloat('word_freq_weight') > 0.0:
            features.append(PointGaussianInverseChiSquaredFeature(
                1, 1, 1, 1, 1))
            weights.append(shared.config['Features'].getfloat('word_freq_weight'))
        if shared.config['Features'].getfloat('word_vec_weight') > 0.0:
            features.append(PointGaussianInverseChiSquaredFeature(
                shared.config['Features'].getint('word_vec_dim'), 10, 0, 10, 0.01))
            weights.append(shared.config['Features'].getfloat('word_vec_weight'))
        result.features = features
        result.weights = weights
        return result

    @staticmethod
    def new_root_feature_set() -> 'PointFeatureSet':
        result = PointFeatureSet()
        features = [AlergiaStringFeature()]
        weights = [1.0]
        if shared.config['Features'].getfloat('word_freq_weight') > 0.0:
            features.append(PointExponentialFeature())
            weights.append(shared.config['Features'].getfloat('word_freq_weight'))
        if shared.config['Features'].getfloat('word_vec_weight') > 0.0:
            features.append(PointGaussianInverseChiSquaredFeature(\
                settings.WORD_VEC_DIM, 10, 0, 10, 1))
            weights.append(shared.config['Features'].getfloat('word_vec_weight'))
        result.features = features
        result.weights = weights
        return result

    @staticmethod
    def new_rule_feature_set():
        result = PointFeatureSet()
        result.features = [UnigramSequenceFeature()]
        result.weights = [1.0]
        return result
    
    def weighted_cost(self, values):
        return sum(w*f.weighted_cost(val) for f, w, val in\
            zip(self.features, self.weights, values))

    def cost(self, values):
        return sum(w*f.cost(val) for f, w, val in\
            zip(self.features, self.weights, values))

    def null_cost(self):
        return sum(w*f.null_cost()\
            for f, w in zip(self.features, self.weights))
    
    def fit(self, values):
        for feature, f_val in zip(self.features, values):
            feature.fit(f_val)

    def weighted_fit(self, values):
        for feature, f_val in zip(self.features, values):
            feature.weighted_fit(f_val)

    def to_string(self):
        return '\t'.join(f.to_string() for f in self.features)

