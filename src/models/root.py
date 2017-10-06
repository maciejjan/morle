import algorithms.alergia
import algorithms.fst
from datastruct.lexicon import LexiconEntry, Lexicon
import shared

import hfst
import numpy as np
from typing import Iterable


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

    def __init__(self) -> None:
#         self.lexicon = lexicon
        self.automaton = hfst.empty_fst()
#         if self.lexicon is None:
#             self.automaton = hfst.empty_fst()
#         else:
#             self.fit()

    # TODO weights are presently ignored, should it be so?!
    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        word_seqs, tag_seqs = [], []
        for entry in lexicon:
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
#         self.recompute_costs()
            
#     def recompute_costs(self) -> None:
#         self.costs = np.empty(len(self.lexicon))
#         for i, entry in enumerate(self.lexicon):
#             self.costs[i] = self.automaton.lookup(entry.symstr)[0][1]

    def root_cost(self, entry :LexiconEntry) -> float:
        return self.automaton.lookup(entry.symstr)[0][1]

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        return np.array([self.root_cost(entry) for entry in lexicon])

    def save(self, filename :str) -> None:
        algorithms.fst.save_transducer(self.automaton, filename)

    @staticmethod
    def load(filename :str) -> 'AlergiaRootModel':
        result = AlergiaRootModel()
        result.automaton = algorithms.fst.load_transducer(filename)
        return result


class RNNRootModel(RootModel):
    pass    # TODO
