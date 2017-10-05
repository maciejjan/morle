import algorithms.fst
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.graph import EdgeSet, GraphEdge
from datastruct.rules import Rule, RuleSet
from models.root import AlergiaRootModel
from models.edge import SimpleEdgeModel
from models.feature import \
     GaussianRootFeatureModel, NeuralRootFeatureModel, RNNRootFeatureModel, \
     GaussianEdgeFeatureModel, NeuralEdgeFeatureModel
from utils.files import file_exists
import shared

import logging
import numpy as np
from typing import Iterable


class ModelSuite:
    # TODO break up in smaller methods
    def __init__(self, rule_set :RuleSet, initialize_models=True) -> None:
        self.rule_set = rule_set
        if initialize_models:
            self.root_model = AlergiaRootModel()
            edge_model_type = shared.config['Models'].get('edge_model')
            if edge_model_type == 'simple':
                self.edge_model = SimpleEdgeModel(rule_set)
            else:
                raise Exception('Unknown edge model: %s' % edge_model_type)
            self.root_feature_model = None
            root_feature_model_type = \
                shared.config['Models'].get('root_feature_model')
            if root_feature_model_type == 'gaussian':
                self.root_feature_model = GaussianRootFeatureModel()
            elif root_feature_model_type == 'neural':
                self.root_feature_model = NeuralRootFeatureModel()
            elif root_feature_model_type == 'rnn':
                lexicon_tr = algorithms.fst.load_transducer(\
                                 shared.filenames['lexicon-tr'])
                alphabet = tuple(sorted(list(lexicon_tr.get_alphabet())))
                longest_paths = lexicon_tr.extract_longest_paths(
                                    max_number=1, output='raw')
                maxlen = len(longest_paths[0][1])
                logging.getLogger('main').debug(\
                    'Detected maximum word length: {}'.format(maxlen))
                self.root_feature_model = RNNRootFeatureModel(alphabet, maxlen)
            elif root_feature_model_type == 'none':
                pass
            else:
                raise Exception('Unknown root feature model: %s' \
                                % edge_model_type)
            self.edge_feature_model = None
            edge_feature_model_type = \
                shared.config['Models'].get('edge_feature_model')
            if edge_feature_model_type == 'gaussian':
                self.edge_feature_model = \
                    GaussianEdgeFeatureModel(self.rule_set)
            elif edge_feature_model_type == 'neural':
                self.edge_feature_model = NeuralEdgeFeatureModel(self.rule_set)
            elif edge_feature_model_type == 'none':
                pass
            else:
                raise Exception('Unknown edge feature model: %s' \
                                % edge_model_type)

    def root_cost(self, entry :LexiconEntry) -> float:
        result = self.root_model.root_cost(entry)
        if self.root_feature_model is not None:
            result += self.root_feature_model.root_cost(entry)
        return result

    def root_costs(self, entries :Iterable[LexiconEntry]) -> np.ndarray:
        result = self.root_model.root_costs(entries)
        if self.root_feature_model is not None:
            result += self.root_feature_model.root_costs(entries)
        return result

    def rule_cost(self, rule :Rule) -> float:
        return self.edge_model.rule_cost(rule)

    def edge_cost(self, edge :GraphEdge) -> float:
        result = self.edge_model.edge_cost(edge)
        if self.edge_feature_model is not None:
            result += self.edge_feature_model.edge_cost(edge)
        return result

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
        rules_file = shared.filenames['rules-modsel']
        if not file_exists(rules_file):
            rules_file = shared.filenames['rules']
        rule_set = RuleSet.load(rules_file)
        result = ModelSuite(rule_set)
        result.root_model = AlergiaRootModel.load(\
                                shared.filenames['root-model'])
        edge_model_type = shared.config['Models'].get('edge_model')
        if edge_model_type == 'simple':
            result.edge_model = SimpleEdgeModel.load(\
                                  shared.filenames['edge-model'], rule_set)
        else:
            raise RuntimeError('Unknown edge model: %s' % edge_model_type)
        result.root_feature_model = None
        root_feature_model_type = \
            shared.config['Models'].get('root_feature_model')
        # load models
        if root_feature_model_type == 'gaussian':
            result.root_feature_model =\
                GaussianRootFeatureModel.load(
                    shared.filenames['root-feature-model'])
        elif root_feature_model_type == 'neural':
            result.root_feature_model =\
                NeuralRootFeatureModel.load(
                    shared.filenames['root-feature-model'])
        elif root_feature_model_type == 'rnn':
            result.root_feature_model =\
                RNNRootFeatureModel.load(
                    shared.filenames['root-feature-model'])
        elif root_feature_model_type == 'none':
            pass
        else:
            raise Exception('Unknown root feature model: %s' \
                            % root_model_type)
        result.edge_feature_model = None
        edge_feature_model_type = \
            shared.config['Models'].get('edge_feature_model')
        if edge_feature_model_type == 'gaussian':
            result.edge_feature_model =\
                GaussianEdgeFeatureModel.load(
                    shared.filenames['edge-feature-model'], self.rule_set)
        elif edge_feature_model_type == 'neural':
            result.edge_feature_model =\
                NeuralEdgeFeatureModel.load(
                    shared.filenames['edge-feature-model'])
        elif edge_feature_model_type == 'none':
            pass
        else:
            raise Exception('Unknown edge feature model: %s' \
                            % edge_model_type)
        return result

