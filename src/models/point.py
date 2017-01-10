from models.generic import Model
from utils.files import *

from collections import defaultdict

class PointModel(Model):

    def __init__(self, lexicon=None, rules=None, rule_domsizes=None):
        self.model_type = 'point'
        Model.__init__(self, lexicon, rules, rule_domsizes)

    def fit_to_lexicon(self, lexicon):
        raise NotImplementedError()

    def fit_to_sample(self, sample):
        def sample_to_edges_by_rule(sample):
            edges_by_rule = defaultdict(lambda: list())
            for edge, weight in sample:
                edges_by_rule[edge.rule].append((edge, weight))
            return edges_by_rule

        edges_by_rule = sample_to_edges_by_rule(sample)
        for rule, edges in edges_by_rule.items():
            self.rule_features[rule].weighted_fit(\
                self.extractor.extract_feature_values_from_weighted_edges(edges))

    def recompute_edge_costs(self, edges):
        for e in edges:
#            logging.getLogger('main').debug(\
#                    ' '.join((e.source.key, e.target.key, str(e.rule),
#                              str(self.edge_cost(e)-e.cost))))
            e.cost = self.edge_cost(e)

    def recompute_root_costs(self, roots):
        for root in roots:
            new_cost = self.rootdist.cost_of_change(\
                    self.extractor.extract_feature_values_from_nodes((root,)), ())
#            logging.getLogger('main').debug(\
#                    ' '.join((root.key,
#                              str(new_cost-root.cost))))
            root.cost = new_cost

    def cost_of_change(self, edges_to_add, edges_to_remove):
        return sum(e.cost for e in edges_to_add) -\
                sum(e.target.cost for e in edges_to_add) -\
                sum(e.cost for e in edges_to_remove) +\
                sum(e.target.cost for e in edges_to_remove)

    def apply_change(self, edges_to_add, edges_to_remove):
#        logging.getLogger('main').debug('change costs = %d - %d + %d - %d' %\
#                (sum(e.cost for e in edges_to_add),
#                 sum(e.cost for e in edges_to_remove),
#                 sum(e.target.cost for e in edges_to_remove),
#                 sum(e.target.cost for e in edges_to_add)))
        self.edges_cost += sum(e.cost for e in edges_to_add) -\
                           sum(e.cost for e in edges_to_remove)
        self.roots_cost += sum(e.target.cost for e in edges_to_remove) -\
                           sum(e.target.cost for e in edges_to_add)

    def edge_cost(self, edge):
        return self.rule_features[edge.rule].cost(\
            self.extractor.extract_feature_values_from_edges((edge,)))

    def save_rules(self, filename):
        with open_to_write(filename) as fp:
            for rule, features in sorted(self.rule_features.items(),
                                         reverse=True,
                                         key=lambda x: x[1][0].trials*x[1][0].prob):
                write_line(fp, (rule, features.to_string()))

