import shared
import pickle
#from models.features.generic import StringFeature
from models.features.extractor import FeatureValueExtractor
from models.features.factory import FeatureSetFactory
#from models.features.marginal import *
#from models.generic import Model
#
#from operator import itemgetter
from collections import defaultdict

class Model:

    def __init__(self, lexicon=None, rules=None, rule_domsizes=None):
        self.extractor = FeatureValueExtractor()
        self.rootdist = None
        self.ruledist = None
        self.rule_features = {}
        self.root_features = None
        self.roots_cost = 0.0
        self.edges_cost = 0.0
        self.rules_cost = 0.0
        if not hasattr(self, 'model_type'):
            self.model_type = 'generic'

        if lexicon:
            self.fit_rootdist(lexicon)
        if rules:
            self.fit_ruledist(rules)
            for rule in rules:
                domsize =\
                    rule_domsizes[rule] if rule_domsizes is not None \
                                           and rule in rule_domsizes \
                    else rule.compute_domsize(lexicon)
                self.add_rule(rule, domsize)
        if lexicon:
            self.add_lexicon(lexicon)
    
    def fit_rootdist(self, lexicon):
        self.rootdist = FeatureSetFactory.new_root_feature_set(self.model_type)
        # fit only the first feature (the MarginalStringFeature)
        self.rootdist[0].fit(\
            self.extractor.extract_feature_values_from_nodes(\
                list(lexicon.iter_nodes()))[0])
    
    def fit_ruledist(self, ruleset):
        self.ruledist = FeatureSetFactory.new_rule_feature_set(self.model_type)
        # fit only the first feature (the MarginalStringFeature)
        self.ruledist[0].fit(\
            self.extractor.extract_feature_values_from_rules(ruleset)[0])

    def add_lexicon(self, lexicon):
        roots_to_add = self.extractor.extract_feature_values_from_nodes(
            lexicon.roots)
        self.roots_cost += self.rootdist.cost_of_change(roots_to_add, [])
        self.rootdist.apply_change(roots_to_add, [])
        for rule, edges in lexicon.edges_by_rule.items():
            edges_to_add = self.extractor\
                               .extract_feature_values_from_edges(edges)
            self.edges_cost += self.rule_features[rule]\
                                   .cost_of_change(edges_to_add, [])
            self.rule_features[rule].fit(edges_to_add)

    def reset(self):
        self.rootdist.reset()
        self.ruledist.reset()
        for features in self.rule_features.values():
            features.reset()
        self.roots_cost = 0.0
        self.edges_cost = sum(f.null_cost() for f in self.rule_features.values())
        self.rules_cost = -self.ruledist.cost_of_change([],\
                            self.extractor.extract_feature_values_from_rules(
                                self.rule_features.keys()))

    def num_rules(self):
        return len(self.rule_features)
    
    def cost(self):
        return self.roots_cost + self.rules_cost + self.edges_cost

    def has_rule(self, rule):
        return rule in self.rule_features
    
    def add_rule(self, rule, domsize):
        self.rule_features[rule] =\
            FeatureSetFactory.new_edge_feature_set(self.model_type, domsize)
        self.rules_cost += self.ruledist.cost_of_change(\
            self.extractor.extract_feature_values_from_rules((rule,)), [])
        self.edges_cost += self.rule_features[rule].null_cost()

    def remove_rule(self, rule):
        self.rules_cost += self.ruledist.cost_of_change([],\
            self.extractor.extract_feature_values_from_rules((rule,)))
        self.edges_cost -= self.rule_features[rule].cost()
        del self.rule_features[rule]

    def rule_cost(self, rule, domsize):
        features = \
            FeatureSetFactory.new_edge_feature_set(self.model_type, domsize)
        return self.ruledist.cost_of_change(\
            self.extractor.extract_feature_values_from_rules((rule,)), []) +\
            features.cost()

    def cost_of_change(self, edges_to_add, edges_to_remove):
        result = 0.0
        root_changes, changes_by_rule =\
            self.extract_feature_values_for_change(
                edges_to_add, edges_to_remove)
        # apply the changes to roots
        roots_to_add, roots_to_remove = root_changes
#        print('rootdist', len(roots_to_add[0]), len(roots_to_remove[0]))
        result += self.rootdist.cost_of_change(
            roots_to_add, roots_to_remove)
        # apply the changes to rule features
        for rule, (values_to_add, values_to_remove) in changes_by_rule.items():
#            print(rule, len(values_to_add[0]), len(values_to_remove[0]))
            result += self.rule_features[rule].cost_of_change(
                values_to_add, values_to_remove)
        return result

    def apply_change(self, edges_to_add, edges_to_remove):
        root_changes, changes_by_rule =\
            self.extract_feature_values_for_change(
                edges_to_add, edges_to_remove)
        # apply the changes to roots
        roots_to_add, roots_to_remove = root_changes
        self.roots_cost += \
            self.rootdist.cost_of_change(roots_to_add, roots_to_remove)
        self.rootdist.apply_change(roots_to_add, roots_to_remove)
        # apply the changes to rule features
        for rule, (values_to_add, values_to_remove) in changes_by_rule.items():
            self.edges_cost += self.rule_features[rule].cost_of_change(
                values_to_add, values_to_remove)
            self.rule_features[rule].apply_change(
                values_to_add, values_to_remove)

    def extract_feature_values_for_change(self, edges_to_add, edges_to_remove):
        # changes to roots
        roots_to_add = self.extractor.extract_feature_values_from_nodes(
            [e.target for e in edges_to_remove])
        roots_to_remove = self.extractor.extract_feature_values_from_nodes(
            [e.target for e in edges_to_add])
        # changes to rule features
        edges_to_add_by_rule = defaultdict(lambda: list())
        edges_to_remove_by_rule = defaultdict(lambda: list())
        rules = set()
        for e in edges_to_add:
            edges_to_add_by_rule[e.rule].append(e)
            rules.add(e.rule)
        for e in edges_to_remove:
            edges_to_remove_by_rule[e.rule].append(e)
            rules.add(e.rule)
        changes_by_rule = {}
        for rule in rules:
            changes_by_rule[rule] = (\
                self.extractor.extract_feature_values_from_edges(\
                    edges_to_add_by_rule[rule]),
                self.extractor.extract_feature_values_from_edges(\
                    edges_to_remove_by_rule[rule]))
        return (roots_to_add, roots_to_remove), changes_by_rule

    def save_to_file(self, filename):
        # forget the transducers, because they are not serializable
        for rule in self.rule_features:
            rule.transducer = None
        with open(shared.options['working_dir'] + filename, 'w+b') as fp:
#            pickle.Pickler(fp, encoding=settings.ENCODING).dump(self)
            pickle.Pickler(fp).dump(self)

    @staticmethod
    def load_from_file(filename):
        with open(shared.options['working_dir'] + filename, 'rb') as fp:
#            pickle.Unpickler(fp, encoding=settings.ENCODING).load()
            return pickle.Unpickler(fp).load()

    def save_rules(self, filename):
        raise NotImplementedError()

    def load_rules(self, filename):
        raise NotImplementedError()

