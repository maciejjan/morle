from models.features.generic import *
import settings

from collections import defaultdict
from scipy.special import betaln

class MarginalFeature(Feature):
    pass

class MarginalStringFeature(MarginalFeature, StringFeature):
    def __init__(self):
        StringFeature.__init__(self)
        self.reset()

    def cost_of_change(self, values_to_add, values_to_remove):
        return sum(self.log_probs[ngr] for val in values_to_add for ngr in val) -\
            sum(self.log_probs[ngr] for val in values_to_remove for ngr in val)

    def apply_change(self, values_to_add, values_to_remove):
        for val in values_to_add:
            for ngr in val:
                self.counts[ngr] += 1
        for val in values_to_remove:
            for ngr in val:
                self.counts[ngr] -= 1

    def cost(self):
        return sum(count*self.log_probs[ngr]\
             for ngr, count in self.counts.items())
    
    def reset(self):
        self.counts = defaultdict(lambda: 0)

class MarginalBinomialFeature(MarginalFeature):
    def __init__(self, trials, alpha=1, beta=1):
        self.trials = trials
        self.count = 0
        self.alpha_0 = alpha
        self.beta_0 = beta
        self.reset()
    
    def cost(self):
        return -betaln(\
                       self.count + self.alpha,\
                       self.trials - self.count + self.beta
                      ) +\
            betaln(self.alpha, self.beta)
    
    def cost_of_change(self, values_to_add, values_to_remove):
        count_change = len(values_to_add) - len(values_to_remove)
        return -betaln(\
                       self.count + count_change + self.alpha,\
                       self.trials - self.count - count_change + self.beta
                      ) +\
                betaln(\
                       self.count + self.alpha,\
                       self.trials - self.count + self.beta
                      )
    
    def apply_change(self, values_to_add, values_to_remove):
        count_change = len(values_to_add) - len(values_to_remove)
        self.count += count_change

    def reset(self):
        self.alpha = self.alpha_0
        self.beta = self.beta_0
    
    def update(self, count):
        self.count = count

class MarginalGaussianInverseChiSquaredFeature(MarginalFeature):
    def __init__(self):
        raise Exception('Not implemented!')

class MarginalFeatureSet(FeatureSet):
    '''Creates sets of features according to the program configuration.'''

    def __init__(self):
        self.features = ()
        self.weights = ()
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    @staticmethod
    def new_edge_feature_set(domsize):
        result = MarginalFeatureSet()
        features = [MarginalBinomialFeature(domsize, alpha=1.1, beta=1.1)]
        weights = [1.0]
        if settings.WORD_FREQ_WEIGHT > 0.0:
            features.append(MarginalGaussianInverseChiSquaredFeature(
                1, 1, 1, 1, 1))
            weights.append(settings.WORD_FREQ_WEIGHT)
        if settings.WORD_VEC_WEIGHT > 0.0:
            features.append(MarginalGaussianInverseChiSquaredFeature(
                settings.WORD_VEC_DIM, 10, 0, 10, 0.01))
            weights.append(settings.WORD_VEC_WEIGHT)
        result.features = tuple(features)
        result.weights = tuple(weights)
        return result

    @staticmethod
    def new_root_feature_set():
        result = MarginalFeatureSet()
        features = [MarginalStringFeature()]
        weights = [1.0]
        if settings.WORD_FREQ_WEIGHT > 0.0:
            features.append(MarginalExponentialFeature())
            weights.append(settings.WORD_FREQ_WEIGHT)
        if settings.WORD_VEC_WEIGHT > 0.0:
#            features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
            features.append(MarginalGaussianInverseChiSquaredFeature(\
                settings.WORD_VEC_DIM, 10, 0, 10, 1))
            weights.append(settings.WORD_VEC_WEIGHT)
        result.features = tuple(features)
        result.weights = tuple(weights)
        return result

    @staticmethod
    def new_rule_feature_set():
        result = MarginalFeatureSet()
        result.features = (MarginalStringFeature(),)
        result.weights = (1.0,)
        return result
    
    def cost(self):
        return sum(w*f.cost(val) for f, w, val in\
            zip(self.features, self.weights, values))

    def cost_of_change(self, values_to_add, values_to_delete):
        # ensure the right size for empty feature vectors
        if not values_to_add:
            values_to_add = ((),) * len(self.features)
        if not values_to_delete:
            values_to_delete = ((),) * len(self.features)
        # apply the change in every single feature
        return sum(f.cost_of_change(values_to_add[i], values_to_delete[i])\
            for i, f in enumerate(self.features))

    def apply_change(self, values_to_add, values_to_delete):
        # ensure the right size for empty feature vectors
        if not values_to_add:
            values_to_add = ((),) * len(self.features)
        if not values_to_delete:
            values_to_delete = ((),) * len(self.features)
        # apply the change in every single feature
        for i, f in enumerate(self.features):
            f.apply_change(values_to_add[i], values_to_delete[i])
    
