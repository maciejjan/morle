from collections import defaultdict

import math

class Feature:
    def __init__(self):
        raise NotImplementedError()
    
    def cost_of_change(self, values_to_add, values_to_remove):
        raise NotImplementedError()
    
    def apply_change(self, values_to_add, values_to_remove):
        raise NotImplementedError()

    def fit(self, values):
        raise NotImplementedError()

    def null_cost(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

class StringFeature(Feature):
    '''A string feature drawn from an n-gram distribution.'''

    def __init__(self):
        self.log_probs = {}

    def cost(self, values):
        return sum(self.smoothing if ngram not in self.log_probs\
            else self.log_probs[ngram]\
            for val in values for ngram in val)

    def fit(self, values):
        counts, total = defaultdict(lambda: 0), 0
        for value in values:
            for ngram in value:
                counts[ngram] += 1
                total += 1
        self.log_probs = {}
        self.smoothing = -math.log(1 / total)
        for ngram, count in counts.items():
            self.log_probs[ngram] = -math.log(count / total)
    
    def num_args(self):
        return 1
    
    def parse_string_args(self, log_probs):
        pass

    def cost_of_change(self, values_to_add, values_to_remove):
        return self.cost(values_to_add) - self.cost(values_to_remove)

    def apply_change(self, values_to_add, values_to_remove):
        pass

    def reset(self):
        pass

class FeatureSet:
    def __init__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    @staticmethod
    def new_edge_feature_set(domsize):
        raise NotImplementedError()

    @staticmethod
    def new_root_feature_set():
        raise NotImplementedError()

    @staticmethod
    def new_rule_feature_set():
        raise NotImplementedError()
    
#    def cost(self):
#        raise NotImplementedError()

    def reset(self):
        for f in self.features:
            f.reset()

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
    
