from models.features.generic import *
import shared

from collections import defaultdict
import numpy as np
from scipy.special import betaln, gammaln
from typing import Any, List


class MarginalFeature(Feature):
    def fit(self, values :Any) -> None:
        self.apply_change(values, [])

#class MarginalStringFeature(MarginalFeature, StringFeature):
#    def __init__(self):
#        StringFeature.__init__(self)
#        self.reset()
#
#    def cost_of_change(self, values_to_add, values_to_remove):
#        return sum(self.log_probs[ngr] for val in values_to_add for ngr in val) -\
#            sum(self.log_probs[ngr] for val in values_to_remove for ngr in val)
#
#    def apply_change(self, values_to_add, values_to_remove):
#        for val in values_to_add:
#            for ngr in val:
#                if ngr not in self.counts:
#                    self.counts[ngr] = 0
#                self.counts[ngr] += 1
#        for val in values_to_remove:
#            for ngr in val:
#                self.counts[ngr] -= 1
#                if self.counts[ngr] == 0:
#                    del self.counts[ngr]
#
#    def cost(self):
#        return sum(count*self.log_probs[ngr]\
#             for ngr, count in self.counts.items())
#    
#    def reset(self):
#        self.counts = {}

class MarginalBinomialFeature(MarginalFeature):
    def __init__(self, trials :int, alpha :float = 1, beta :float = 1) -> None:
        self.trials = trials
        self.count = 0
        self.alpha_0 = alpha
        self.beta_0 = beta
        self.reset()

    def null_cost(self) -> float:
        return -betaln(\
                       self.alpha_0,\
                       self.trials + self.beta_0
                      ) +\
            betaln(self.alpha_0, self.beta_0)

    def empty(self) -> int:
        return 0
    
    def cost(self) -> float:
        return -betaln(\
                       self.count + self.alpha,\
                       self.trials - self.count + self.beta
                      ) +\
            betaln(self.alpha, self.beta)
    
    def cost_of_change(self, values_to_add :int, 
                             values_to_remove :int) -> float:
#         count_change = len(values_to_add) - len(values_to_remove)
        count_change = values_to_add - values_to_remove
        return -betaln(\
                       self.count + count_change + self.alpha,\
                       self.trials - self.count - count_change + self.beta
                      ) +\
                betaln(\
                       self.count + self.alpha,\
                       self.trials - self.count + self.beta
                      )
    
    def apply_change(self, values_to_add :int, values_to_remove :int) -> None:
#         count_change = len(values_to_add) - len(values_to_remove)
        count_change = values_to_add - values_to_remove
        self.count += count_change

    def reset(self) -> None:
        self.alpha = self.alpha_0
        self.beta = self.beta_0
    
# TODO deprecated
#     def update(self, count :int) -> None:
#         self.count = count


class MarginalExponentialFeature(MarginalFeature):
    def __init__(self):
        raise NotImplementedError()


class MarginalGaussianInverseChiSquaredFeature(MarginalFeature):
    def __init__(self, dim, kappa_0, mu_0, nu_0, var_0):
        self.dim = dim
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0 * np.ones(dim)
        self.nu_0 = nu_0
        self.var_0 = var_0 * np.ones(dim)
        self.reset()

    def null_cost(self):
        return self.cost_with_parameters(self.kappa_0, self.mu_0, self.nu_0, self.var_0)

    def cost(self):
        return self.cost_with_parameters(self.kappa_n, self.mu_n, self.nu_n, self.var_n)

    def cost_of_change(self, values_to_add, values_to_remove):
#        print(len(values_to_add), len(values_to_remove))
        s_0, s_1, s_2 = self.statistics_for_change(values_to_add, values_to_remove)
        kappa, mu, nu, var = self.parameters_from_statistics(s_0, s_1, s_2)
        return self.cost_with_parameters(kappa, mu, nu, var) - self.cost()

    def apply_change(self, values_to_add, values_to_remove):
        s_0, s_1, s_2 = self.statistics_for_change(values_to_add, values_to_remove)
        kappa, mu, nu, var = self.parameters_from_statistics(s_0, s_1, s_2)
        self.s_0 = s_0
        self.s_1 = s_1
        self.s_2 = s_2
        self.kappa_n = kappa
        self.mu_n = mu
        self.nu_n = nu
        self.var_n = var
        
    def reset(self):
        self.s_0 = np.zeros(1)
        self.s_1 = np.zeros(self.dim)
        self.s_2 = np.zeros(self.dim)
        self.kappa_n = self.kappa_0
        self.mu_n = self.mu_0
        self.nu_n = self.nu_0
        self.var_n = self.var_0

    def update(self):
        raise NotImplementedError()

    def statistics_for_change(self, values_to_add, values_to_remove):
        def statistics_from_values(values):
            if not values:
                return 0, np.zeros(self.dim), np.zeros(self.dim)
            m = np.stack(values)
            return len(values), np.sum(m, axis=0), np.sum(m**2, axis=0)

#        print('MarginalGaussianInverseChiSquaredFeature', len(values_to_add), len(values_to_remove))
        s_0_add, s_1_add, s_2_add = statistics_from_values(values_to_add)
        s_0_rmv, s_1_rmv, s_2_rmv = statistics_from_values(values_to_remove)
        s_0 = self.s_0 + s_0_add - s_0_rmv
#        print(s_0)
        if s_0 == 0:
            return s_0, np.zeros(self.dim), np.zeros(self.dim)
        else:
            s_1 = (self.s_1*self.s_0 + s_1_add - s_1_rmv) / s_0
            s_2 = (self.s_2*self.s_0 + s_2_add - s_2_rmv) / s_0
            return s_0, s_1, s_2

    def parameters_from_statistics(self, s_0, s_1, s_2):
        kappa = self.kappa_0 + s_0
        if kappa == 0:
            raise RuntimeError('kappa = 0!')
        mu = (self.kappa_0*self.mu_0 + s_0*s_1) / kappa
        nu = self.nu_0 + s_0
        var = (self.nu_0*self.var_0 + s_0*s_2 - s_0*s_1**2 +\
            (s_0*self.kappa_0/(self.kappa_0+s_0))*(self.mu_0-s_1)**2) / nu
        return kappa, mu, nu, var

    def cost_with_parameters(self, kappa, mu, nu, var):
        # TODO optimize - a vector of coordinate costs, sum at the end
        def coordinate_cost(kappa, mu, nu, var, kappa_0, mu_0, nu_0, var_0):
            return -gammaln(nu/2)+gammaln(nu_0/2) - 0.5*np.log(kappa_0) +\
                0.5*np.log(kappa) - 0.5*nu_0*np.log(nu_0*var_0) +\
                0.5*nu*np.log(nu*var) + 0.5*self.s_0*np.log(np.pi)
        
        return np.asscalar(
                sum(coordinate_cost(kappa, mu[i], nu, var[i],\
                                       self.kappa_0, self.mu_0[i], self.nu_0, self.var_0[i])\
                       for i in range(self.dim)))
        

class MarginalFeatureSet(FeatureSet):
    '''Creates sets of features according to the program configuration.'''

    def __init__(self):
        self.features = ()
        self.weights = ()
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    @staticmethod
    def new_edge_feature_set(domsize) -> 'MarginalFeatureSet':
        result = MarginalFeatureSet()
        features = [MarginalBinomialFeature(domsize, alpha=1.1, beta=1.1)]
        weights = [1.0]
        if shared.config['Features'].getfloat('word_freq_weight') > 0.0:
            features.append(MarginalGaussianInverseChiSquaredFeature(
                1, 1, 1, 1, 1))
            weights.append(shared.config['Features'].getfloat('word_freq_weight'))
        if shared.config['Features'].getfloat('word_vec_weight') > 0.0:
            features.append(MarginalGaussianInverseChiSquaredFeature(
                shared.config['Features'].getint('word_vec_dim'), 1, 0, 1, 0.01))
            weights.append(shared.config['Features'].getfloat('word_vec_weight'))
        result.features = tuple(features)
        result.weights = tuple(weights)
        return result

    @staticmethod
    def new_root_feature_set() -> 'MarginalFeatureSet':
        result = MarginalFeatureSet()
#        features = [MarginalStringFeature()]
        features = [AlergiaStringFeature()]
        weights = [1.0]
        if shared.config['Features'].getfloat('word_freq_weight') > 0.0:
#            features.append(MarginalExponentialFeature())
            features.append(MarginalGaussianInverseChiSquaredFeature(\
                1, 1, 1, 1, 1))
            weights.append(shared.config['Features'].getfloat('word_freq_weight'))
        if shared.config['Features'].getfloat('word_vec_weight') > 0.0:
#            features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
            features.append(MarginalGaussianInverseChiSquaredFeature(\
                shared.config['Features'].getint('word_vec_dim'), 1, 0, 1, 1))
            weights.append(shared.config['Features'].getfloat('word_vec_weight'))
        result.features = tuple(features)
        result.weights = tuple(weights)
        return result

    @staticmethod
    def new_rule_feature_set() -> 'MarginalFeatureSet':
        result = MarginalFeatureSet()
        result.features = (UnigramSequenceFeature(),)
#         result.features = (ZeroCostFeature(),)
        result.weights = (1.0,)
        return result
    
#     def cost(self):
#         return sum(w*f.cost() for f, w in\
#             zip(self.features, self.weights))
# 
#     def reset(self):
#         for f in self.features:
#             f.reset()
# 
#     def cost_of_change(self, values_to_add :List,
#                              values_to_delete :List) -> float:
#         # ensure the right size for empty feature vectors
#         if not values_to_add:
#             values_to_add = ((),) * len(self.features)
#         if not values_to_delete:
#             values_to_delete = ((),) * len(self.features)
# #        print('MarginalFeatureSet.cost_of_change', len(values_to_add), len(values_to_delete))
# #        print('MarginalFeatureSet.cost_of_change', len(values_to_add[0]), len(values_to_delete[0]))
#         # apply the change in every single feature
#         return sum(f.cost_of_change(values_to_add[i], values_to_delete[i])\
#             for i, f in enumerate(self.features))
# 
#     def apply_change(self, values_to_add :List,
#                            values_to_delete :List) -> None:
#         # ensure the right size for empty feature vectors
#         if not values_to_add:
#             values_to_add = ((),) * len(self.features)
#         if not values_to_delete:
#             values_to_delete = ((),) * len(self.features)
#         # apply the change in every single feature
#         for i, f in enumerate(self.features):
#             f.apply_change(values_to_add[i], values_to_delete[i])
#     
