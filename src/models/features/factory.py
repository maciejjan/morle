from models.features.generic import *
from models.features.marginal import *
from models.features.point import *

class FeatureSetFactory:
    @staticmethod
    def new_edge_feature_set(model_type, domsize) -> FeatureSet:
        if model_type == 'generic':
            return FeatureSet.new_edge_feature_set(domsize)
        elif model_type == 'point':
            return PointFeatureSet.new_edge_feature_set(domsize)
        elif model_type == 'marginal':
            return MarginalFeatureSet.new_edge_feature_set(domsize)

    @staticmethod
    def new_root_feature_set(model_type) -> FeatureSet:
        if model_type == 'generic':
            return FeatureSet.new_root_feature_set()
        elif model_type == 'point':
            return PointFeatureSet.new_root_feature_set()
        elif model_type == 'marginal':
            return MarginalFeatureSet.new_root_feature_set()

    @staticmethod
    def new_rule_feature_set(model_type) -> FeatureSet:
        if model_type == 'generic':
            return FeatureSet.new_rule_feature_set()
        elif model_type == 'point':
            return PointFeatureSet.new_rule_feature_set()
        elif model_type == 'marginal':
            return MarginalFeatureSet.new_rule_feature_set()

