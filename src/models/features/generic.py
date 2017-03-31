import algorithms.alergia
import algorithms.fst
from datastruct.lexicon import tokenize_word
import shared

from collections import defaultdict
import hfst
import logging
import math
import numpy as np
from typing import Any, Iterable, List


class Feature:
    def __init__(self) -> None:
        raise NotImplementedError()
    
    def cost_of_change(self, values_to_add :Any, 
                             values_to_remove :Any) -> float:
        raise NotImplementedError()
    
    def apply_change(self, values_to_add :Any, 
                           values_to_remove :Any) -> None:
        raise NotImplementedError()

    def empty(self) -> None:
        raise NotImplementedError()

    def fit(self, values :Any) -> None:
        raise NotImplementedError()

    def null_cost(self) -> float:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()


class NumericFeature(Feature):
    def cost_of_change(self, values_to_add :np.ndarray, 
                             values_to_remove :np.ndarray) -> float:
        raise NotImplementedError()
    
    def apply_change(self, values_to_add :np.ndarray, 
                           values_to_remove :np.ndarray) -> None:
        raise NotImplementedError()

    def fit(self, values :np.ndarray) -> None:
        raise NotImplementedError()

    def empty(self) -> np.ndarray:
        return np.array([])


class SequenceValuedFeature(Feature):
    def cost_of_change(self, values_to_add :Iterable[List],
                             values_to_remove :Iterable[List]) -> float:
        raise NotImplementedError()
    
    def apply_change(self, values_to_add :Iterable[List], 
                           values_to_remove :Iterable[List]) -> None:
        raise NotImplementedError()

    def fit(self, values :Iterable[List]) -> None:
        raise NotImplementedError()

    def empty(self) -> Iterable[List]:
        return []


class StringValuedFeature(Feature):
    def cost_of_change(self, values_to_add :Iterable[str],
                             values_to_remove :Iterable[str]) -> float:
        raise NotImplementedError()
    
    def apply_change(self, values_to_add :Iterable[str],
                           values_to_remove :Iterable[str]) -> None:
        raise NotImplementedError()

    def fit(self, values :Iterable[str]) -> None:
        raise NotImplementedError()

    def empty(self) -> str:
        return ''


class ZeroCostFeature(Feature):
    def __init__(self) -> None:
        pass
    
    def cost_of_change(self, values_to_add :Any, 
                             values_to_remove :Any) -> None:
        return 0.0
    
    def apply_change(self, values_to_add :Any, 
                           values_to_remove :Any) -> None:
        pass

    def fit(self, values :Any) -> None:
        pass

    def null_cost(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass

# TODO StringFeature -> UnigramFeature
# new class: StringValuedFeature for string-valued features
# also: FloatValuedFeature etc.
# the existence of edges moved to an extra feature!
class UnigramSequenceFeature(SequenceValuedFeature):
    '''A sequence feature drawn from a unigram distribution.'''

    def __init__(self) -> None:
        self.log_probs = {}
        self.smoothing = 0.0

    def cost(self, values :Iterable[List]) -> float:
        return sum(self.smoothing if ngram not in self.log_probs\
            else self.log_probs[ngram]\
            for val in values for ngram in val)

    def fit(self, values :Iterable[List]) -> float:
        counts, total = defaultdict(lambda: 0), 0
        for value in values:
            for ngram in value:
                counts[ngram] += 1
                total += 1
        self.log_probs = {}
        self.smoothing = -math.log(1 / total)
        for ngram, count in counts.items():
            self.log_probs[ngram] = -math.log(count / total)
            
    def null_cost(self) -> float:
        return 0.0
    
    def cost_of_change(self, values_to_add :Iterable[List],
                             values_to_remove :Iterable[List]) -> float:
        return self.cost(values_to_add) - self.cost(values_to_remove)

    def apply_change(self, values_to_add :Iterable[List],
                           values_to_remove :Iterable[List]) -> None:
        pass

    def reset(self) -> None:
        pass

# TODO
class AlergiaStringFeature(StringValuedFeature):
    '''A string feature learnt with the ALERGIA algorithm.'''

    def __init__(self) -> None:
        self.automaton = None
        self.cache = {}
        # TODO remove cache as soon as the memory leak in HFST is fixed

    def cost(self, values :Iterable[str]) -> float:
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

    def fit(self, values :Iterable[str]) -> None:
        # TODO code duplicated from modules.compile -> refactor!!!
        word_seqs, tag_seqs = [], []
        for val in values:
            word, tag, disamb = tokenize_word(val)
            word_seqs.append(word)
            tag_seqs.append(tag)

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
#         self.automaton.minimize()
        self.automaton.convert(hfst.ImplementationType.HFST_OLW_TYPE)
#         algorithms.fst.save_transducer(self.automaton, 'stringfeat.fsm',
#                                        type=hfst.HFST_OLW_TYPE)
            
    def null_cost(self) -> float:
        return 0.0
    
    def cost_of_change(self, values_to_add :Iterable[str], 
                             values_to_remove :Iterable[str]) -> float:
        return self.cost(values_to_add) - self.cost(values_to_remove)

    def apply_change(self, values_to_add :Iterable[str], 
                           values_to_remove :Iterable[str]) -> float:
        pass

    def reset(self) -> None:
        pass

class FeatureSet:
    def __init__(self) -> None:
        raise NotImplementedError()
    
    def __getitem__(self, idx) -> Feature:
        return self.features[idx]
    
    @staticmethod
    def new_edge_feature_set(domsize) -> 'FeatureSet':
        raise NotImplementedError()

    @staticmethod
    def new_root_feature_set() -> 'FeatureSet':
        raise NotImplementedError()

    @staticmethod
    def new_rule_feature_set() -> 'FeatureSet':
        raise NotImplementedError()
    
#    def cost(self):
#        raise NotImplementedError()

    def reset(self) -> None:
        for f in self.features:
            f.reset()

    def null_cost(self):
        return sum(f.null_cost() for f in self.features)

    def cost_of_change(self, values_to_add :List,
                             values_to_delete :List) -> float:
        # ensure the right size for empty feature vectors
        if not values_to_add:
            values_to_add = [feature.empty() for feature in self.features]
        if not values_to_delete:
            values_to_delete = [feature.empty() for feature in self.features]
        # apply the change in every single feature
        return sum(f.cost_of_change(values_to_add[i], values_to_delete[i])\
            for i, f in enumerate(self.features))

    def apply_change(self, values_to_add :List,
                           values_to_delete :List) -> None:
        # ensure the right size for empty feature vectors
        if not values_to_add:
            values_to_add = [feature.empty() for feature in self.features]
#             values_to_add = ((),) * len(self.features)
        if not values_to_delete:
            values_to_delete = [feature.empty() for feature in self.features]
#             values_to_delete = ((),) * len(self.features)
        # apply the change in every single feature
        for i, f in enumerate(self.features):
            f.apply_change(values_to_add[i], values_to_delete[i])
    
