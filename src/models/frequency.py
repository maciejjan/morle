from datastruct.lexicon import LexiconEntry, Lexicon
from models.generic import Model, ModelFactory, UnknownModelTypeException

import numpy as np


class RootFrequencyModel(Model):
    pass


class ZipfRootFrequencyModel(RootFrequencyModel):
    def __init__(self) -> None:
        pass

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        pass

#     def root_cost(self, entry :LexiconEntry) -> float:
#         return float(self.roots_cost([entry]))
# 
    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        freqs = np.array([entry.freq for entry in lexicon])
        return -np.log(1/(freqs * (freqs+1)))

    def save(self, filename :str) -> None:
        pass


class RootFrequencyModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str) -> RootFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'zipf':
            return ZipfRootFrequencyModel()
        else:
            raise UnknownModelTypeException('root frequency', model_type)

    @staticmethod
    def load(model_type :str, filename :str) -> RootFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'zipf':
            return ZipfRootFrequencyModel()
        else:
            raise UnknownModelTypeException('root frequency', model_type)


class EdgeFrequencyModel(Model):
    pass


class LogNormalEdgeFrequencyModel(EdgeFrequencyModel):
    def __init__(self) -> None:
        pass

    def fit(self, edge_set :EdgeSet, weights :np.ndarray) -> None:
        pass

    def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()


class EdgeFrequencyModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str) -> EdgeFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'lognormal':
            return LogNormalEdgeFrequencyModel()
        else:
            raise UnknownModelTypeException('edge frequency', model_type)

    @staticmethod
    def load(model_type :str, filename :str) -> EdgeFeatureModel:
        if model_type == 'none':
            return None
        elif model_type == 'lognormal':
            return LogNormalFrequencyModel.load(filename)
        else:
            raise UnknownModelTypeException('edge frequency', model_type)
