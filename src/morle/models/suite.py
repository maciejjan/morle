import algorithms.fst
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.graph import EdgeSet, GraphEdge, FullGraph
from datastruct.rules import Rule, RuleSet
from models.root import RootModelFactory
from models.rule import RuleModelFactory
from models.tag import TagModelFactory
from models.edge import EdgeModelFactory
from models.feature import RootFeatureModelFactory, EdgeFeatureModelFactory
from models.frequency import RootFrequencyModelFactory, \
                             EdgeFrequencyModelFactory
from utils.files import file_exists
import shared

import logging
import numpy as np
from typing import Iterable, List, Union


class ModelSuite:

    def __init__(self, rule_set :RuleSet,
                 lexicon :Lexicon = None,
                 initialize_models :bool = True) -> None:
        self.rule_set = rule_set
        self.added_root_cost = shared.config['Models']\
                                     .getfloat('added_root_cost')
        self.added_rule_cost = shared.config['Models']\
                                     .getfloat('added_rule_cost')
        if initialize_models:
            self.rule_model = RuleModelFactory.create(
                                  shared.config['Models'].get('rule_model'))
            self.root_model = RootModelFactory.create(
                                  shared.config['Models'].get('root_model'))
            self.edge_model = EdgeModelFactory.create(
                                  shared.config['Models'].get('edge_model'),
                                  self.rule_set)
            self.root_tag_model = \
                TagModelFactory.create(
                    shared.config['Models'].get('root_tag_model'))
            self.root_frequency_model = \
                RootFrequencyModelFactory.create(
                    shared.config['Models'].get('root_frequency_model'))
            self.root_feature_model = \
                RootFeatureModelFactory.create(
                    shared.config['Models'].get('root_feature_model'),
                    lexicon)
            self.edge_frequency_model = \
                EdgeFrequencyModelFactory.create(
                    shared.config['Models'].get('edge_frequency_model'),
                    self.rule_set)
            self.edge_feature_model = \
                EdgeFeatureModelFactory.create(
                    shared.config['Models'].get('edge_feature_model'),
                    self.rule_set)
            self.frequency_weight = \
                shared.config['Features'].getfloat('word_freq_weight')
            self.feature_weight = \
                shared.config['Features'].getfloat('word_vec_weight')

    def initialize(self, graph :FullGraph) -> None:
        '''Fit the models assuming unit weights for all roots and edges'''
        root_weights = None
        if shared.config['General'].getboolean('supervised'):
            # supervised learning -- take only nodes with no incoming edges
            # as roots
            root_weights = np.array([1 if not graph.predecessors(node) else 0 \
                                     for node in graph.lexicon])
        else:
            root_weights = np.ones(len(graph.lexicon))
        edge_weights = np.ones(len(graph.edge_set))
        self.root_model.fit(graph.lexicon, root_weights)
        if self.rule_model is not None:
            self.rule_model.fit(self.rule_set)
        self.fit(graph.lexicon, graph.edge_set, root_weights, edge_weights)

    def root_cost(self, entry :LexiconEntry) -> float:
        result = self.root_model.root_cost(entry)
        if self.root_tag_model is not None:
            result += self.root_tag_model.root_cost(entry)
        if self.root_frequency_model is not None:
            result += self.frequency_weight * \
                      self.root_frequency_model.root_cost(entry)
        if self.root_feature_model is not None:
            result += self.feature_weight * \
                      self.root_feature_model.root_cost(entry)
        result += self.added_root_cost
        return result

    def roots_cost(self, entries :Union[LexiconEntry, Iterable[LexiconEntry]]) -> np.ndarray:
        result = self.root_model.root_costs(entries)
        if self.root_tag_model is not None:
            result += self.root_tag_model.root_costs(entries)
        if self.root_frequency_model is not None:
            result += self.frequency_weight * \
                      self.root_frequency_model.root_costs(entries)
        if self.root_feature_model is not None:
            result += self.feature_weight * \
                      self.root_feature_model.root_costs(entries)
        result += self.added_root_cost
        return result

    def rule_cost(self, rule :Rule) -> float:
        result = 0.0
        if self.rule_model is not None:
            result += self.rule_model.rule_cost(rule)
        result += self.edge_model.rule_cost(rule)
        result += self.added_rule_cost
        return result
# 
#     def edge_cost(self, edge :GraphEdge) -> float:
#         result = self.edge_model.edge_cost(edge)
#         if self.edge_feature_model is not None:
#             result += self.edge_feature_model.edge_cost(edge)
#         return result

    # TODO EdgeSet -> Iterable[Edge]?

    def edges_prob(self, edges :EdgeSet) -> np.ndarray:
        return self.edge_model.edges_prob(edges)

    def edge_cost(self, edge :GraphEdge) -> float:
        result = self.edge_model.edge_cost(edge)
        if self.edge_frequency_model is not None:
            result += self.frequency_weight * \
                      self.edge_frequency_model.edge_cost(edge)
        if self.edge_feature_model is not None:
            result += self.feature_weight * \
                      self.edge_feature_model.edge_cost(edge)
        return result
    
    def edges_cost(self, edges :EdgeSet) -> np.ndarray:
        result = self.edge_model.edges_cost(edges)
        if self.edge_frequency_model is not None:
            result += self.frequency_weight * \
                      self.edge_frequency_model.edges_cost(edges)
        if self.edge_feature_model is not None:
            result += self.feature_weight * \
                      self.edge_feature_model.edges_cost(edges)
        return result

    def null_cost(self) -> float:
        return self.edge_model.null_cost()

    def predict_target_feature_vec(self, edge :GraphEdge) -> np.ndarray:
        if self.edge_feature_model is not None:
            return self.edge_feature_model.predict_target_feature_vec(edge)
        else:
            raise Exception('No edge feature model!')

#     def edges_cost(self, edge_set :EdgeSet) -> np.ndarray:
#         # TODO cost of an edge set -- optimized computation
#         raise NotImplementedError()

    def iter_rules(self) -> Iterable[Rule]:
        return iter(self.rule_set)

    def delete_rules(self, rules_to_delete :List[Rule]) -> None:
        raise NotImplementedError()
        
    def fit(self, lexicon :Lexicon, edge_set :EdgeSet, 
            root_weights :np.ndarray, edge_weights :np.ndarray) -> None:
        self.edge_model.fit(edge_set, edge_weights)
        if self.root_tag_model is not None:
            self.root_tag_model.fit(lexicon, root_weights)
        if self.root_frequency_model is not None:
            self.root_frequency_model.fit(lexicon, root_weights)
        if self.root_feature_model is not None:
            self.root_feature_model.fit(lexicon, root_weights)
        if self.edge_frequency_model is not None:
            self.edge_frequency_model.fit(edge_set, edge_weights)
        if self.edge_feature_model is not None:
            self.edge_feature_model.fit(edge_set, edge_weights)


    def save(self) -> None:
        self.root_model.save(shared.filenames['root-model'])
        self.edge_model.save(shared.filenames['edge-model'])
        if self.rule_model is not None:
            self.rule_model.save(shared.filenames['rule-model'])
        if self.root_tag_model is not None:
            self.root_tag_model.save(shared.filenames['root-tag-model'])
        if self.root_frequency_model is not None:
            self.root_frequency_model.save(shared.filenames['root-frequency-model'])
        if self.root_feature_model is not None:
            self.root_feature_model.save(shared.filenames['root-feature-model'])
        if self.edge_frequency_model is not None:
            self.edge_frequency_model.save(shared.filenames['edge-frequency-model'])
        if self.edge_feature_model is not None:
            self.edge_feature_model.save(shared.filenames['edge-feature-model'])

    @staticmethod
    def is_loadable() -> bool:
        return file_exists(shared.filenames['root-model']) and \
               file_exists(shared.filenames['edge-model']) and \
               (shared.config['Models'].get('root_tag_model') == 'none' or \
                file_exists(shared.filenames['root-tag-model'])) and \
               (shared.config['Models'].get('root_frequency_model') == 'none' or \
                file_exists(shared.filenames['root-frequency-model'])) and \
               (shared.config['Models'].get('root_feature_model') == 'none' or \
                file_exists(shared.filenames['root-feature-model'])) and \
               (shared.config['Models'].get('edge_frequency_model') == 'none' or \
                file_exists(shared.filenames['edge-frequency-model'])) and \
               (shared.config['Models'].get('edge_feature_model') == 'none' or \
                file_exists(shared.filenames['edge-feature-model']))

    @staticmethod
    def load() -> 'ModelSuite':
        rules_file = shared.filenames['rules-modsel']
        if not file_exists(rules_file):
            rules_file = shared.filenames['rules']
        rule_set = RuleSet.load(rules_file)
        lexicon = Lexicon.load(shared.filenames['wordlist'])
        result = ModelSuite(rule_set)
        result.rule_model = RuleModelFactory.load(
                                shared.config['Models'].get('rule_model'),
                                shared.filenames['rule-model'])
        result.root_model = RootModelFactory.load(
                                shared.config['Models'].get('root_model'),
                                shared.filenames['root-model'])
        result.edge_model = EdgeModelFactory.load(
                                shared.config['Models'].get('edge_model'),
                                shared.filenames['edge-model'],
                                rule_set)
        result.root_tag_model = \
            TagModelFactory.load(
                shared.config['Models'].get('root_tag_model'),
                shared.filenames['root-tag-model'])
        result.root_frequency_model = \
            RootFrequencyModelFactory.load(
                shared.config['Models'].get('root_frequency_model'),
                shared.filenames['root-frequency-model'])
        result.root_feature_model = \
            RootFeatureModelFactory.load(
                shared.config['Models'].get('root_feature_model'),
                shared.filenames['root-feature-model'],
                lexicon)
        result.edge_frequency_model = \
            EdgeFrequencyModelFactory.load(
                shared.config['Models'].get('edge_frequency_model'),
                shared.filenames['edge-frequency-model'],
                rule_set)
        result.edge_feature_model = \
            EdgeFeatureModelFactory.load(
                shared.config['Models'].get('edge_feature_model'),
                shared.filenames['edge-feature-model'],
                rule_set)
        return result

