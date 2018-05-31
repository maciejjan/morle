import algorithms.alergia
import algorithms.fst
from datastruct.lexicon import LexiconEntry, Lexicon
from models.generic import Model, ModelFactory, UnknownModelTypeException
import shared

import hfst
import numpy as np
from typing import Iterable


class RootModel(Model):
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
        self.automaton = hfst.empty_fst()

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

    def root_cost(self, entry :LexiconEntry) -> float:
        lookup_results = self.automaton.lookup(entry.symstr)
        if not lookup_results:
            return np.inf
        return lookup_results[0][1]

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        return np.array([self.root_cost(entry) for entry in lexicon])

    def save(self, filename :str) -> None:
        algorithms.fst.save_transducer(self.automaton, filename)

    @staticmethod
    def load(filename :str) -> 'AlergiaRootModel':
        result = AlergiaRootModel()
        result.automaton = algorithms.fst.load_transducer(filename)
        return result


class NGramRootModel(RootModel):
    pass    # TODO


class RNNRootModel(RootModel):
    pass    # TODO


class RootModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str) -> RootModel:
        if model_type == 'alergia':
            return AlergiaRootModel()
        else:
            raise UnknownModelTypeException('root', model_type)

    @staticmethod
    def load(model_type :str, filename :str) -> RootModel:
        if model_type == 'alergia':
            return AlergiaRootModel.load(filename)
        else:
            raise UnknownModelTypeException('root', model_type)

