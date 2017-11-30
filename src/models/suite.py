import algorithms.fst
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.graph import EdgeSet, GraphEdge
from datastruct.rules import Rule, RuleSet
from models.root import RootModelFactory
from models.edge import EdgeModelFactory
from models.feature import RootFeatureModelFactory, EdgeFeatureModelFactory
from utils.files import file_exists
import shared

import logging
import numpy as np
from typing import Iterable, Union


class ModelSuite:

    def __init__(self, rule_set :RuleSet,
                 lexicon :Lexicon = None,
                 initialize_models :bool = True) -> None:
        self.rule_set = rule_set
        if initialize_models:
            self.root_model = RootModelFactory.create(
                                  shared.config['Models'].get('root_model'))
            self.edge_model = EdgeModelFactory.create(
                                  shared.config['Models'].get('edge_model'),
                                  self.rule_set)
            self.root_feature_model = \
                RootFeatureModelFactory.create(
                    shared.config['Models'].get('root_feature_model'),
                    lexicon)
            self.edge_feature_model = \
                EdgeFeatureModelFactory.create(
                    shared.config['Models'].get('edge_feature_model'),
                    self.rule_set)

    def root_cost(self, entry :LexiconEntry) -> float:
        result = self.root_model.root_cost(entry)
        if self.root_feature_model is not None:
            result += self.root_feature_model.root_cost(entry)
        return result

    def roots_cost(self, entries :Union[LexiconEntry, Iterable[LexiconEntry]]) -> np.ndarray:
        result = self.root_model.root_costs(entries)
        if self.root_feature_model is not None:
            result += self.root_feature_model.root_costs(entries)
        return result

    def rule_cost(self, rule :Rule) -> float:
        return self.edge_model.rule_cost(rule)
# 
#     def edge_cost(self, edge :GraphEdge) -> float:
#         result = self.edge_model.edge_cost(edge)
#         if self.edge_feature_model is not None:
#             result += self.edge_feature_model.edge_cost(edge)
#         return result

    # TODO EdgeSet -> Iterable[Edge]?
    def edges_cost(self, edges :EdgeSet) -> np.ndarray:
        result = self.edge_model.edges_cost(edges)
        if self.edge_feature_model is not None:
            result += self.edge_feature_model.edges_cost(edges)
        return result

    def null_cost(self) -> float:
        return self.edge_model.null_cost()

#     def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
#         # TODO cost of an edge set -- optimized computation
#         raise NotImplementedError()

    def iter_rules(self) -> Iterable[Rule]:
        return iter(self.rule_set)
        
    def fit(self, lexicon :Lexicon, edge_set :EdgeSet, 
            root_weights :np.ndarray, edge_weights :np.ndarray) -> None:
        self.edge_model.fit(edge_set, edge_weights)
        if self.root_feature_model is not None:
            self.root_feature_model.fit(lexicon, root_weights)
        if self.edge_feature_model is not None:
            self.edge_feature_model.fit(edge_set, edge_weights)


    def save(self) -> None:
        self.root_model.save(shared.filenames['root-model'])
        self.edge_model.save(shared.filenames['edge-model'])
        if self.root_feature_model is not None:
            self.root_feature_model.save(shared.filenames['root-feature-model'])
        if self.edge_feature_model is not None:
            self.edge_feature_model.save(shared.filenames['edge-feature-model'])

    @staticmethod
    def is_loadable() -> bool:
        return file_exists(shared.filenames['root-model']) and \
               file_exists(shared.filenames['edge-model']) and \
               (shared.config['Features'].getfloat('word_vec_weight') == 0 or \
                file_exists(shared.filenames['feature-model']))

    @staticmethod
    def load() -> 'ModelSuite':
        rule_set = RuleSet.load(shared.filenames['rules'])
        lexicon = Lexicon.load(shared.filenames['wordlist'])
        result = ModelSuite(rule_set)
        result.root_model = RootModelFactory.load(
                                shared.config['Models'].get('root_model'),
                                shared.filenames['root-model'])
        result.edge_model = EdgeModelFactory.load(
                                shared.config['Models'].get('edge_model'),
                                shared.filenames['edge-model'],
                                rule_set)
        result.root_feature_model = \
            RootFeatureModelFactory.load(
                shared.config['Models'].get('root_feature_model'),
                shared.filenames['root-feature-model'],
                lexicon)
        result.edge_feature_model = \
            EdgeFeatureModelFactory.load(
                shared.config['Models'].get('edge_feature_model'),
                shared.filenames['edge-feature-model'],
                rule_set)
        return result

