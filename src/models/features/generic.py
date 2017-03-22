import algorithms.alergia
import algorithms.fst
from datastruct.lexicon import tokenize_word
import shared

from collections import defaultdict
import hfst
import logging
import math
import numpy as np

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

class ZeroCostFeature:
    def __init__(self):
        pass
    
    def cost_of_change(self, values_to_add, values_to_remove):
        return 0.0
    
    def apply_change(self, values_to_add, values_to_remove):
        pass

    def fit(self, values):
        pass

    def null_cost(self):
        return 0.0

    def reset(self):
        pass

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
            
    def null_cost(self):
        return 0
    
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

# TODO
class AlergiaStringFeature(Feature):
    '''A string feature learnt with the ALERGIA algorithm.'''

    def __init__(self):
        self.automaton = None
        self.cache = {}
        # TODO remove cache as soon as the memory leak in HFST is fixed

    def cost(self, values):
        result = 0.0
        for val in values:
            try:
                if val not in self.cache:
                    self.cache[val] = self.automaton.lookup(val)[0][1]
                result += self.cache[val]
            except IndexError:
                logging.getLogger('main').warning(
                    'Zero root probability for: {}'.format(val))
                return np.inf   # infinite cost if some string is impossible
        return result

    def fit(self, values):
        # TODO code duplicated from modules.compile -> refactor!!!
        word_seqs, tag_seqs = [], []
        for val in values:
            word, tag, disamb = tokenize_word(val)
#             print(word, tag, disamb)
            word_seqs.append(word)
            tag_seqs.append(tag)
#         word_seqs = [(val.word, 1) for val in values]
#         tag_seqs = [(val.tag, 1) for val in values]

#         word_pta = algorithms.alergia.prefix_tree_acceptor(word_seqs)
        alpha = shared.config['compile'].getfloat('alergia_alpha')
        freq_threshold = shared.config['compile'].getint('alergia_freq_threshold')
        self.automaton = \
            algorithms.alergia.alergia(word_seqs, alpha=alpha, 
                                       freq_threshold=freq_threshold)

#         tag_automaton = hfst.HfstTransducer(
#                           algorithms.alergia.normalize_weights(
#                             algorithms.alergia.prefix_tree_acceptor(tag_seqs)))
        tag_automaton = \
            algorithms.alergia.prefix_tree_acceptor(tag_seqs).to_hfst()
        tag_automaton.minimize()

#         self.automaton = hfst.HfstTransducer(automaton)
        self.automaton.concatenate(tag_automaton)
        self.automaton.remove_epsilons()
#         self.automaton.minimize()
        self.automaton.convert(hfst.ImplementationType.HFST_OLW_TYPE)
#         algorithms.fst.save_transducer(self.automaton, 'stringfeat.fsm',
#                                        type=hfst.HFST_OLW_TYPE)
            
    def null_cost(self):
        return 0
    
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

    def null_cost(self):
        return sum(f.null_cost() for f in self.features)

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
    
