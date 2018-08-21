import algorithms.alergia
import algorithms.fst
from datastruct.lexicon import LexiconEntry, Lexicon
from models.generic import Model, ModelFactory, UnknownModelTypeException
from utils.files import open_to_write, read_tsv_file, write_line
import shared

from collections import defaultdict
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
#         word_seqs, tag_seqs = [], []
        word_seqs = []
        for entry in lexicon:
            word_seqs.append(entry.word)
#             tag_seqs.append(entry.tag)

        alpha = shared.config['compile'].getfloat('alergia_alpha')
        freq_threshold = \
            shared.config['compile'].getint('alergia_freq_threshold')
        self.automaton = \
            algorithms.alergia.alergia(word_seqs, alpha=alpha, 
                                       freq_threshold=freq_threshold).to_hfst()
#         tag_automaton = \
#             algorithms.alergia.prefix_tree_acceptor(tag_seqs).to_hfst()
#         tag_automaton.minimize()

#         self.automaton.concatenate(tag_automaton)
        self.automaton.remove_epsilons()
        self.automaton.convert(hfst.ImplementationType.HFST_OLW_TYPE)

    def root_cost(self, entry :LexiconEntry) -> float:
#         lookup_results = self.automaton.lookup(entry.symstr)
        lookup_results = self.automaton.lookup(''.join(entry.word))
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


class UnigramRootModel(RootModel):
    UNKNOWN = '<UNKN>'

    def __init__(self) -> None:
        self.probs = { UnigramRootModel.UNKNOWN : 1 }

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        counts = defaultdict(lambda: 0)
        for i, entry in enumerate(lexicon):
            for sym in entry.word:
                counts[sym] += weights[i]
        counts[UnigramRootModel.UNKNOWN] += 1
        total = sum(counts.values())
        self.probs = {}
        for sym, count in counts.items():
            self.probs[sym] = count / total

    def root_cost(self, entry :LexiconEntry) -> float:
        result = 1.0
        for sym in entry.word:
            result *= self.probs[sym] \
                      if sym in self.probs \
                      else self.probs[UnigramRootModel.UNKNOWN]
        return result

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        result = np.ones(len(lexicon))
        for i, entry in enumerate(lexicon):
            result[i] = self.root_cost(entry)
        return result

    def save(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for sym, prob in self.probs.items():
                write_line(fp, (sym, prob))

    @staticmethod
    def load(filename :str) -> 'UnigramRootModel':
        result = UnigramRootModel()
        for sym, prob in read_tsv_file(filename, types=(str, float)):
            result.probs[sym] = prob
        return result


class NGramRootModel(RootModel):
    pass    # TODO


class RNNRootModel(RootModel):

    def __init__(self) -> None:
        raise NotImplementedError()

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        # TODO determine the alphabet from the lexicon
        # TODO count the occurrences of each symbol
        # TODO normalize the symbol probabilities
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'AlergiaRootModel':
        raise NotImplementedError()


class RootModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str) -> RootModel:
        if model_type == 'alergia':
            return AlergiaRootModel()
        elif model_type == 'rnn':
            return RNNRootModel()
        elif model_type == 'unigram':
            return UnigramRootModel()
        else:
            raise UnknownModelTypeException('root', model_type)

    @staticmethod
    def load(model_type :str, filename :str) -> RootModel:
        if model_type == 'alergia':
            return AlergiaRootModel.load(filename)
        elif model_type == 'rnn':
            return RNNRootModel.load(filename)
        elif model_type == 'unigram':
            return UnigramRootModel.load(filename)
        else:
            raise UnknownModelTypeException('root', model_type)

