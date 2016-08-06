import math
import random
import sys
import time
from operator import itemgetter
import numpy as np
from scipy.special import beta, betaln, expit
from datastruct.lexicon import *
from datastruct.rules import *
from models.point import PointModel
from models.marginal import MarginalModel
from utils.files import *
from utils.printer import *
import settings

# TODO optimizing the rule set
# - instead choosing rules uniformly, choose according to the percentage of graphs without this rule
#   - or the best option: choose according to the subsample size obtained by choosing this rule?
#     (how to compute it quickly?)
# - if the subsample size falls below 1% of the sample size -> resample?

# TODO check which statistic require starting at a null graph

# Beta distribution parameters
ALPHA = 1
BETA = 1

NUM_WARMUP_ITERATIONS = 1000000
NUM_ITERATIONS = 10000000
NUM_RULE_ITERATIONS = 10000
ANNEALING_TEMPERATURE = 50

### STATISTICS ###

class MCMCStatistic:
    def __init__(self, sampler):        raise Exception('Not implemented!')
    def reset(self, sampler):            raise Exception('Not implemented!')
    def update(self, sampler):            raise Exception('Not implemented!')
    def edge_added(self, sampler):        raise Exception('Not implemented!')
    def edge_removed(self, sampler):    raise Exception('Not implemented!')
    def next_iter(self, sampler):        raise Exception('Not implemented!')

class ScalarStatistic(MCMCStatistic):
    def __init__(self, sampler):
        self.reset(sampler)
    
    def reset(self, sampler):
        self.val = 0
        self.last_modified = 0

    def update(self, sampler):
        raise Exception('Not implemented!')

    def edge_added(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def edge_removed(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def next_iter(self, sampler):
        pass
    
    def value(self):
        return self.val

class ExpectedLogLikelihoodStatistic(ScalarStatistic):
    def update(self, sampler):
        pass
    
    def edge_added(self, sampler, idx, edge):
        pass

    def edge_removed(self, sampler, idx, edge):
        pass
    
    def next_iter(self, sampler):
        self.val = (self.val * (sampler.num-1) + sampler.logl()) / sampler.num

class TimeStatistic(ScalarStatistic):
    def reset(self, sampler):
        self.started = time.time()
        self.val = 0
    
    def update(self, sampler):
        self.val = time.time() - self.started
    
    def edge_added(self, sampler, idx, edge):
        pass

    def edge_removed(self, sampler, idx, edge):
        pass

# TODO
class GraphsWithoutRuleSetStatistic(ScalarStatistic):
    def __init__(self, sampler, forbidden_rules):
        self.forbidden_rules = forbidden_rules
        self.reset(sampler)

    def reset(self, sampler):
        self.val = 0
        self.last_modified = 0
        self.forbidden_edges =\
            sum(count for rule, count in sampler.lexicon.rules_c.items()\
                if rule in self.forbidden_rules)

    def update(self, sampler):
        self.val =\
            (self.val * self.last_modified +\
             int(self.forbidden_edges == 0) *\
                (sampler.num - self.last_modified)) /\
            sampler.num
        self.last_modified = sampler.num

    def edge_added(self, sampler, idx, edge):
        if edge.rule in self.forbidden_rules:
            if self.forbidden_edges == 0:
                self.update(sampler)
            self.forbidden_edges += 1

    def edge_removed(self, sampler, idx, edge):
        if edge.rule in self.forbidden_rules:
            if self.forbidden_edges == 1:
                self.update(sampler)
            self.forbidden_edges -= 1

    def next_iter(self, sampler):
        pass

class AcceptanceRateStatistic(ScalarStatistic):
    def update(self, sampler):
        pass
    
    def edge_added(self, sampler, idx, edge):
        self.acceptance(sampler)

    def edge_removed(self, sampler, idx, edge):
        self.acceptance(sampler)

    def acceptance(self, sampler):
        if sampler.num > self.last_modified:
            self.val = (self.val * self.last_modified + 1) / sampler.num
            self.last_modified = sampler.num

class EdgeStatistic(MCMCStatistic):
    def __init__(self, sampler):
        self.reset(sampler)

    def reset(self, sampler):
        self.values = [0] * sampler.len_edges
        self.last_modified = [0] * sampler.len_edges
    
    def update(self, sampler):
        raise Exception('Not implemented!')

    def edge_added(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def edge_removed(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def next_iter(self, sampler):
        pass
    
    def value(self, idx):
        return self.values[idx]

class EdgeFrequencyStatistic(EdgeStatistic):
    def update(self, sampler):
        for i, edge in enumerate(sampler.edges):
            if edge in edge.source.edges:
                # the edge was present in the last graphs
                self.edge_removed(sampler, i, edge)
            else:
                # the edge was absent in the last graphs
                self.edge_added(sampler, i, edge)

    def edge_added(self, sampler, idx, edge):
        self.values[idx] =\
            self.values[idx] * self.last_modified[idx] / sampler.num
        self.last_modified[idx] = sampler.num

    def edge_removed(self, sampler, idx, edge):
        self.values[idx] =\
            (self.values[idx] * self.last_modified[idx] +\
             (sampler.num - self.last_modified[idx])) /\
            sampler.num
        self.last_modified[idx] = sampler.num

class RuleStatistic(MCMCStatistic):
    def __init__(self, sampler):
        self.values = {}
        self.last_modified = {}
        self.reset(sampler)

    def reset(self, sampler):
#        for rule in sampler.model.rules:
        for rule in sampler.model.rule_features:
            self.values[rule] = 0.0
            self.last_modified[rule] = 0
    
    def update(self, sampler):
#        for rule in sampler.model.rules:
        for rule in sampler.model.rule_features:
            self.update_rule(rule, sampler)
    
    def update_rule(self, rule, sampler):
        raise Exception('Not implemented!')
    
    def edge_added(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def edge_removed(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def next_iter(self, sampler):
        pass
    
    def value(self, rule):
        return self.values[rule]

class RuleFrequencyStatistic(RuleStatistic):
    def update_rule(self, rule, sampler):
        self.values[rule] = \
            (self.values[rule] * self.last_modified[rule] +\
             sampler.lexicon.rules_c[rule] * (sampler.num - self.last_modified[rule])) /\
            sampler.num
        self.last_modified[rule] = sampler.num
    
    def edge_added(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)

    def edge_removed(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)

# TODO
class RuleExpectedContributionStatistic(RuleStatistic):
    def update_rule(self, rule, sampler):
        if self.last_modified[rule] < sampler.num:
            edges = sampler.lexicon.edges_by_rule[rule]
            new_value = sampler.model.cost_of_change([], edges)
            self.values[rule] = \
                (self.values[rule] * self.last_modified[rule] +\
                 new_value * (sampler.num - self.last_modified[rule])) /\
                sampler.num
            self.last_modified[rule] = sampler.num
    
    def edge_added(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)

    def edge_removed(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)

# TODO
class RuleChangeCountStatistic(RuleStatistic):
    def reset(self, sampler):
        for rule in sampler.lexicon.ruleset.keys():
            self.values[rule] = 0
            self.last_modified[rule] = 0

    def update_rule(self, rule, sampler):
        pass
    
    def edge_added(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 1:
            self.values[rule] += 1

    def edge_removed(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 0:
            self.values[rule] += 1

# TODO
class RuleGraphsWithoutStatistic(RuleStatistic):
    def update_rule(self, rule, sampler):
        if sampler.lexicon.rules_c[rule] > 0:
            self.values[rule] = \
                self.values[rule] * self.last_modified[rule] / sampler.num
            self.last_modified[rule] = sampler.num
        else:
            self.values[rule] = \
                (self.values[rule] * self.last_modified[rule] +\
                 sampler.num - self.last_modified[rule]) / sampler.num
    
    def edge_added(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 1:
            self.values[rule] = \
                (self.values[rule] * self.last_modified[rule] +\
                 sampler.num - self.last_modified[rule]) / sampler.num
            self.last_modified[rule] = sampler.num

    def edge_removed(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 0:
            self.values[rule] = \
                self.values[rule] * self.last_modified[rule] / sampler.num
            self.last_modified[rule] = sampler.num


# TODO
class RuleIntervalsWithoutStatistic(MCMCStatistic):
    def __init__(self, sampler):
        self.intervals = {}
        self.int_start = {}
        self.reset(sampler)

    def reset(self, sampler):
        for rule in sampler.lexicon.ruleset.keys():
            self.intervals[rule] = []
            if sampler.lexicon.rules_c[rule] > 0:
                self.int_start[rule] = None
            else:
                self.int_start[rule] = 0

    def update(self, sampler):
        for rule in sampler.lexicon.ruleset.keys():
            if self.int_start[rule] is not None:
                self.intervals[rule].append((self.int_start[rule], sampler.num))
                self.int_start[rule] = None
    
    def edge_added(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 1:
            if self.int_start[rule] is None:
                raise Exception('Interval with no left end: %s' % rule)
            self.intervals[rule].append((self.int_start[rule], sampler.num))
            self.int_start[rule] = None

    def edge_removed(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 0:
            self.int_start[rule] = sampler.num
    
    def next_iter(self, sampler):
        pass

### SAMPLERS ###

# TODO monitor the number of moves from each variant and their acceptance rates!
# TODO refactor
class MCMCGraphSampler:
    def __init__(self, model, lexicon, edges):
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.edges_hash = {}
        for idx, e in enumerate(edges):
            self.edges_hash[(e.source, e.target)] = (idx, e)
        self.len_edges = len(edges)
        self.num = 0        # iteration number
        self.stats = {}
#        self.accept_all = False
    
    def add_stat(self, name, stat):
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def logl(self):
        if isinstance(self.model, PointModel):
            return self.lexicon.cost
        elif isinstance(self.model, MarginalModel):
            return self.model.cost()
        else:
            raise Exception('Unknown model type.')

    def run_sampling(self):
        print('Warming up the sampler...')
        pp = progress_printer(settings.SAMPLING_WARMUP_ITERATIONS)
        for i in pp:
            self.next()
        self.reset()
        pp = progress_printer(settings.SAMPLING_ITERATIONS)
        print('Sampling...')
        for i in pp:
            self.next()
        self.update_stats()

    def next(self):
        def consider_adding_edge(edge):
#            return math.exp(-self.model.edge_cost(edge) + edge.target.cost)
            return math.exp(-edge.cost + edge.target.cost)

        def consider_removing_edge(edge):
#            return math.exp(-edge.target.cost + self.model.edge_cost(edge))
            return math.exp(-edge.target.cost + edge.cost)

        # increase the number of iterations
        self.num += 1

        # select an edge randomly
        edge_idx = random.randrange(self.len_edges)            
        edge = self.edges[edge_idx]

        # determine the changes to the lexicon
        edges_to_add, edges_to_remove = [], []
        if edge.target.parent == edge.source:
            # delete the selected edge
            edges_to_remove.append(edge)
        else:
            # add the selected edge
            edges_to_add.append(edge)
            if edge.source.has_ancestor(edge.target):    # cycle
                # swap parents
                # determine the nodes relevant for swap operation
                node_1, node_2 = edge.source, edge.target
                node_3 = node_2.parent\
                         if node_2.parent is not None\
                         else None
                node_4 = node_1.parent
                node_5 = node_4
                if node_5 != node_2:
                    while node_5.parent != node_2: 
                        node_5 = node_5.parent
                # check whether all the relevant edges exist
                edge_2_5, edge_3_1, edge_3_2, edge_3_5, edge_4_1 =\
                    None, None, None, None, None
                try:
                    edge_2_5 = self.edges_hash[(node_2, node_5)][1]
                    edge_3_1 = self.edges_hash[(node_3, node_1)][1]
                    edge_3_2 = self.edges_hash[(node_3, node_2)][1]
                    edge_3_5 = self.edges_hash[(node_3, node_5)][1]
                    edge_4_1 = self.edges_hash[(node_4, node_1)][1]
                except KeyError:
                    self.num -= 1
                    return
                # choose the variant of the swap move and perform it
                if random.random() < 0.5:
                    edges_to_remove.append(edge_3_2)
                    edges_to_remove.append(edge_4_1)
                    edges_to_add.append(edge_3_1)
                else:
                    edges_to_remove.append(edge_2_5)
                    edges_to_remove.append(edge_3_2)
                    edges_to_add.append(edge_3_5)
            elif edge.target.parent is not None:        # word_2 already has a parent
                # delete the current parent of word_2
                node_2, node_3 = edge.target, edge.target.parent
                edge_3_2 = self.edges_hash[(node_3, node_2)][1]
                edges_to_remove.append(edge_3_2)

        # compute the probability ratio of the proposed graph
        # related to the current one
        prob_ratio = None
#        if self.accept_all:
#            prob_ratio = 1.0
        if isinstance(self.model, PointModel):
            prob_ratio = 1.0
            for e in edges_to_add:
                prob_ratio *= consider_adding_edge(e)
            for e in edges_to_remove:
                prob_ratio *= consider_removing_edge(e)
        elif isinstance(self.model, MarginalModel):
#            print()
#            print(self.model.rootdist.features[1].s_0)
#            for e in edges_to_add:
#                print('Adding: %s -> %s via %s' % (e.source.key, e.target.key, str(e.rule)))
#                print(self.model.rule_features[e.rule][1].s_0)
#            for e in edges_to_remove:
#                print('Removing: %s -> %s via %s' % (e.source.key, e.target.key, str(e.rule)))
#            print('cost_of_change = %f' % self.model.cost_of_change(edges_to_add, edges_to_remove))
#            print('acc_prob = %f' % prob_ratio)
            prob_ratio = math.exp(\
                -self.model.cost_of_change(edges_to_add, edges_to_remove))
        else:
            raise Exception('Unknown model type.')

        # accept/reject
        if prob_ratio >= 1 or prob_ratio >= random.random():
#            print('Accepted')
            # remove edges and update stats
            for e in edges_to_remove:
#                r = self.lexicon[w_2].deriving_rule()
                idx = self.edges_hash[(e.source, e.target)][0]
                self.lexicon.remove_edge(e)
                if isinstance(self.model, MarginalModel):
                    self.model.apply_change([], [e])
                for stat in self.stats.values():
                    stat.edge_removed(self, idx, e)
            # add edges and update stats
            for e in edges_to_add:
                idx = self.edges_hash[(e.source, e.target)][0]
                self.lexicon.add_edge(e)
                if isinstance(self.model, MarginalModel):
                    self.model.apply_change([e], [])
                for stat in self.stats.values():
                    stat.edge_added(self, idx, e)

        # update the remaining stats
        for stat in self.stats.values():
            stat.next_iter(self)
    
    def reset(self):
        self.num = 0
        for stat in self.stats.values():
            stat.reset(self)

    def update_stats(self):
        for stat in self.stats.values():
            stat.update(self)

# TODO use simulated annealing
class MCMCRuleSampler:
    def __init__(self, model, lexicon, edges):
        self.num = 0
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.full_ruleset = set(model.rule_features)
        self.ruleset = self.full_ruleset
        self.rule_domsize = {}
        self.rule_costs = {}
        for rule in self.full_ruleset:
            self.rule_domsize[rule] = self.model.rule_features[rule][0].trials
            self.rule_costs[rule] = self.model.rule_cost(rule, self.rule_domsize[rule])
        self.cost, self.rule_contrib = \
            self.evaluate_proposal(self.ruleset)

    def next(self):
        next_ruleset = self.propose_next_ruleset()
        self.print_proposal(next_ruleset)
        cost, next_rule_contrib = self.evaluate_proposal(next_ruleset)
        acc_prob = 1 if cost < self.cost else \
            math.exp((self.cost - cost) * ((self.num+10) / 40)) *\
            proposal_probability_ratio(next_ruleset, next_rule_contrib)
        if random.random() < acc_prob:
            self.cost = cost
            self.rule_contrib = next_rule_contrib
            self.accept_ruleset(next_ruleset)
            print('accepted')
        self.num += 1

    def propose_next_ruleset(self):
        result = set()
        for rule in self.full_ruleset:
            if random.random() < self.rule_prob(rule):
                result.add(rule)
        return result

    def evaluate_proposal(self, ruleset):
        self.model.reset()
        new_model = MarginalModel(None, None)
        new_model.rootdist = self.model.rootdist
        new_model.ruledist = self.model.ruledist
        new_model.roots_cost = self.model.roots_cost
        for rule in ruleset:
            new_model.add_rule(rule, self.rule_domsize[rule])
        print(new_model.roots_cost, new_model.rules_cost, new_model.edges_cost)

        self.lexicon.reset(new_model)
        graph_sampler = MCMCGraphSampler(new_model, self.lexicon,\
            [edge for edge in self.edges if edge.rule in ruleset])
        graph_sampler.add_stat('logl', ExpectedLogLikelihoodStatistic(graph_sampler))
        graph_sampler.add_stat('contrib', RuleExpectedContributionStatistic(graph_sampler))
        graph_sampler.run_sampling()

        return graph_sampler.stats['logl'].val, graph_sampler.stats['contrib'].values

    def accept_ruleset(self, new_ruleset):
        for rule in self.ruleset - new_ruleset:
            self.model.remove_rule(rule)
        for rule in new_ruleset - self.ruleset:
            self.model.add_rule(rule, self.rule_domsize[rule])
        self.ruleset = new_ruleset

    def print_proposal(self, new_ruleset):
        for rule in self.ruleset - new_ruleset:
            print('delete: %s' % str(rule))
        for rule in new_ruleset - self.ruleset:
            print('restore: %s' % str(rule))

class RuleSetProposalDistribution:
    def __init__(self, rule_contrib, rule_costs, num):
        self.rule_prob = {}
        for rule in rule_costs:
            rule_score = -rule_costs[rule] +\
                (rule_contrib[rule] if rule in rule_contrib else 0)
            self.rule_prob[rule] =\
                expit(rule_score * ((num+10) / 40))

    def propose(self):
        next_ruleset = set()
        for rule, prob in self.rule_prob.items():
            if random.random() < prob:
                next_ruleset.add(rule)
        return next_ruleset

    def proposal_logprob(self, ruleset):
        return sum(\
                   (np.log(prob) if rule in ruleset else np.log(1-prob)) \
                   for rule, prob in self.rule_prob.items())


class MCMCAnnealingRuleSampler:
    def __init__(self, model, lexicon, edges):
        self.num = 0
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.full_ruleset = set(model.rule_features)
        self.ruleset = self.full_ruleset
        self.rule_domsize = {}
        self.rule_costs = {}
        for rule in self.full_ruleset:
            self.rule_domsize[rule] = self.model.rule_features[rule][0].trials
            self.rule_costs[rule] = self.model.rule_cost(rule, self.rule_domsize[rule])
        self.cost, self.proposal_dist = \
            self.evaluate_proposal(self.ruleset)

    def next(self):
        next_ruleset = self.proposal_dist.propose()
        self.print_proposal(next_ruleset)
        cost, next_proposal_dist = self.evaluate_proposal(next_ruleset)
        acc_prob = 1 if cost < self.cost else \
            math.exp((self.cost - cost) * ((self.num+10) / 40)) *\
            math.exp(next_proposal_dist.proposal_logprob(self.ruleset) -\
                     self.proposal_dist.proposal_logprob(next_ruleset))
        print('acc_prob = %f' % acc_prob)
        if random.random() < acc_prob:
            self.cost = cost
            self.proposal_dist = next_proposal_dist
            self.accept_ruleset(next_ruleset)
            print('accepted')
        else:
            print('rejected')
        self.num += 1

    def evaluate_proposal(self, ruleset):
#        self.model.reset()
        new_model = MarginalModel(None, None)
        new_model.rootdist = self.model.rootdist
        new_model.ruledist = self.model.ruledist
#        new_model.roots_cost = self.model.roots_cost
        for rule in ruleset:
            new_model.add_rule(rule, self.rule_domsize[rule])
        new_model.reset()
        self.lexicon.reset(new_model)
        new_model.fit_to_lexicon(self.lexicon)
        print(new_model.roots_cost, new_model.rules_cost, new_model.edges_cost, new_model.cost())

        graph_sampler = MCMCGraphSampler(new_model, self.lexicon,\
            [edge for edge in self.edges if edge.rule in ruleset])
        graph_sampler.add_stat('logl', ExpectedLogLikelihoodStatistic(graph_sampler))
        graph_sampler.add_stat('contrib', RuleExpectedContributionStatistic(graph_sampler))
        graph_sampler.run_sampling()

        return graph_sampler.stats['logl'].val,\
            RuleSetProposalDistribution(
                graph_sampler.stats['contrib'].values,
                self.rule_costs, self.num)

    def accept_ruleset(self, new_ruleset):
        for rule in self.ruleset - new_ruleset:
            self.model.remove_rule(rule)
        for rule in new_ruleset - self.ruleset:
            self.model.add_rule(rule, self.rule_domsize[rule])
        self.ruleset = new_ruleset

    def print_proposal(self, new_ruleset):
        for rule in self.ruleset - new_ruleset:
            print('delete: %s' % str(rule))
        for rule in new_ruleset - self.ruleset:
            print('restore: %s' % str(rule))


class MCMCRulePosteriorSampler:
    def __init__(self, model, lexicon, edges):
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.full_ruleset = set(self.model.rule_features)
        self.curr_ruleset = self.full_ruleset
        self.rule_costs = {}
        self.rule_domsize = {}
        for rule in self.full_ruleset:
            self.rule_domsize[rule] = self.model.rule_features[rule][0].trials
            self.rule_costs[rule] = self.model.rule_cost(rule, self.rule_domsize[rule])

    def next(self):

        def compute_rule_contributions(model):
            graph_sampler = MCMCGraphSampler(model, self.lexicon,\
                [edge for edge in self.edges if edge.rule in model.rule_features])
            graph_sampler.add_stat('rule_contrib',\
                RuleExpectedContributionStatistic(graph_sampler))
            graph_sampler.run_sampling()
            result = {}
            for rule, cost in self.rule_costs.items():
                result[rule] = -cost +\
                    (graph_sampler.stats['rule_contrib'].value(rule)\
                     if model.has_rule(rule) else 0)
            return result

        def propose_next_ruleset(contrib):
            return set(rule for rule, c in contrib.items() \
                if random.random() < expit(c))

        def compute_ruleset_weights(excluded_rules_1, excluded_rules_2):
            graph_sampler = MCMCGraphSampler(self.model, self.lexicon, self.edges)
            graph_sampler.add_stat('weight_1',\
                GraphsWithoutRuleSetStatistic(graph_sampler, excluded_rules_1))
            graph_sampler.add_stat('weight_2',\
                GraphsWithoutRuleSetStatistic(graph_sampler, excluded_rules_2))
            graph_sampler.run_sampling()
            return graph_sampler.stats['weight_1'].value(),\
                graph_sampler.stats['weight_2'].value()

        def compute_rule_prob_logratio(excluded_rules_1, excluded_rules_2):
            return sum(self.model.rule_cost(rule) for rule in excluded_rules_2) -\
                sum(self.model.rule_cost(rul) for rule in excluded_rules_1)

        def compute_lex_prob_logratio(weight_1, excluded_rules_1,\
                                      weight_2, excluded_rules_2):
            null_betaln = betaln(ALPHA, BETA)
            return sum(betaln(ALPHA, BETA + self.rule_domsize[rule])-null_betaln \
                for rule in excluded_rules_1) + math.log(weight_1) -\
                sum(betaln(ALPHA, BETA + self.rule_domsize[rule])-null_betaln \
                    for rule in excluded_rules_2) - math.log(weight_2)

        def compute_proposal_prob_logratio(contrib_1, ruleset_1, contrib_2, ruleset_2):
            return sum(math.log(expit(contrib_2[rule])) for rule in ruleset_1) -\
                sum(math.log(expit(contrib_1[rule])) for rule in ruleset_2)

        def print_proposal(ruleset_1, ruleset_2):
            for rule in ruleset_1 - ruleset_2:
                print('delete: %s' % str(rule))
            for rule in ruleset_2 - ruleset_1:
                print('restore: %s' % str(rule))

        print('num_rules = %d' % self.model.num_rules())
        # first sampling: compute the the expected contributions of rules
        # (and save them to a file)
        rule_contrib = compute_rule_contributions(self.model)
        # propose the next step according to the contributions
        next_ruleset = propose_next_ruleset(rule_contrib)
        print_proposal(self.curr_ruleset, next_ruleset)
        # create a model which contains the sum of old and new rule set
        combined_ruleset = self.curr_ruleset | next_ruleset
        self.model.reset()
        self.lexicon.reset(self.model)
        for rule in combined_ruleset - self.curr_ruleset:
            self.model.add_rule(rule, self.domsizes[rule])
        # compute the contributions according to the proposed model
        next_model = MarginalModel(None, None)
        next_model.rootdist = self.model.rootdist
        next_model.ruledist = self.model.ruledist
        for rule in next_ruleset:
            next_model.add_rule(rule, self.rule_domsize[rule])
        next_contrib = compute_rule_contributions(next_model)
        # second sampling: compute the expected number of graphs without
        #   rules not appearing in the first and in the second set
        curr_excluded_rules = combined_ruleset - self.curr_ruleset
        next_excluded_rules = combined_ruleset - next_ruleset
        curr_ruleset_weight, next_ruleset_weight =\
            compute_ruleset_weights(curr_excluded_rules, next_excluded_rules)
        if curr_ruleset_weight <= 0 or next_ruleset_weight <= 0:
            print('zero weights: %f, %f' % (curr_ruleset_weight, next_ruleset_weight))
            return # TODO exception? 
        # compute the three relevant values for acceptance/rejection:
        #   1. rule probability ratio
        #   2. lexicon probability ratio (from sampling)
        #   3. proposal probability ratio
        rule_prob_logratio =\
            compute_rule_prob_logratio(self.curr_ruleset, next_ruleset)
        lex_prob_logratio =\
            compute_lex_prob_ratio(curr_ruleset_weight,
                                   curr_excluded_rules,
                                   next_ruleset_weight,
                                   next_excluded_rules
            )
        proposal_prob_logratio = \
            compute_proposal_prob_logratio(rule_contrib,
                                           self.curr_ruleset,
                                           next_contrib,
                                           next_ruleset)
        # accept/reject
        acc_prob = math.exp(rule_prob_logratio + lex_prob_logratio +\
                            proposal_prob_logratio)
        print('acc_prob = %f' % acc_prob)
        self.model.reset()
        rules_to_remove = set()
        if acc_prob >= 1 or acc_prob >= random.random():
            rules_to_remove = combined_ruleset - next_ruleset
            self.curr_ruleset = next_ruleset
            print('accepted')
        else:
            rules_to_remove = combined_ruleset - self.curr_ruleset
            print('rejected')
        for rule in rules_to_remove:
            self.model.remove_rule(rule)
        # TODO if the likelihood is greater than the previous maximum:
        #   save the rule set
        #   (what about saving the lexicon?)


        # TODO store the rule frequency in the samples as a statistic



#class MCMCRuleSetSampler:
#    #TODO: sample rules: choose a rule to delete and sample graphs to determine acceptance/rejection
#    def __init__(self, sample, intervals, ruleset, starting_temp):
#        self.sample = sample        # list of log-likelihoods (with a full rule set)
#        self.sample_len = len(sample)
#        self.intervals = intervals
#        self.ruleset = ruleset
#        self.rules_list = list(self.ruleset.keys())
#        self.len_ruleset = len(ruleset)
#        self.deleted_rules = set()
#        self.best_set = None                    # store the best set of rules
#        self.logl = sum(sample)/self.sample_len    # current log-likelihood
#        self.starting_temp = starting_temp
#        self.temperature = starting_temp
#        self.subsample_size = self.sample_len
#        self.best_deleted_rules = set()
#        self.best_logl = self.logl
#        self.num = 0
#    
#    def next(self, logfp):
#        def rule_zero_frequency_prob(rule):
#            return beta(ALPHA, self.ruleset[rule].domsize + BETA) / beta(ALPHA, BETA)
#
#        def merge_two_intervals(int_1, int_2):
#            i, j = 0, 0
#            result = []
#            if not int_1 or not int_2: return result
#            a_1, b_1 = int_1[0]
#            a_2, b_2 = int_2[0]
#            while i < len(int_1) and j < len(int_2):
#                if b_1 <= a_2:
#                    i += 1
#                    if i == len(int_1): break
#                    a_1, b_1 = int_1[i]
#                    continue
#                elif b_2 <= a_1:
#                    j += 1
#                    if j == len(int_2): break
#                    a_2, b_2 = int_2[j]
#                    continue
#                else:
#                    if b_1 > b_2:
#                        result.append((max(a_1, a_2), b_2))
#                        j += 1
#                        if j == len(int_2): break
#                        a_2, b_2 = int_2[j]
#                        continue
#                    else:
#                        result.append((max(a_1, a_2), b_1))
#                        if b_1 == b_2:
#                            j += 1
#                            if j == len(int_2): break
#                            a_2, b_2 = int_2[j]
#                        i += 1
#                        if i == len(int_1): break
#                        a_1, b_1 = int_1[i]
#                        continue
#            return result
#
#        def merge_n_intervals(ints):
#            result = [(0, self.sample_len)]
#            for interval in ints:
#                result = merge_two_intervals(result, interval)
#            return result
#
#        # TODO what if deleted_rules is empty?
#        def likelihood(deleted_rules):
#            # compute likelihood with the set of deleted rules from the sample
#            result, sum_weights = 0.0, 0.0
#            weight, log_weight = 1.0, 0.0
#            log_prior_weight = 0.0
#            for rule in deleted_rules:
#                rzfp = rule_zero_frequency_prob(rule)
##                weight /= rzfp
#                log_weight += math.log(rzfp)
#                log_prior_weight += math.log(self.ruleset.rsp.rule_prob(rule))
##            print(weight)
#            intervals = merge_n_intervals([self.intervals[r] for r in deleted_rules])
#            if not intervals:
#                return -float('inf')
#            for int_begin, int_end in intervals:
#                for logl in self.sample[int_begin:int_end]:
#                    result += logl - log_weight - log_prior_weight
#                    sum_weights += 1
##            if sum_weights < 0.01 * NUM_ITERATIONS:
##                return -float('inf')
#            result /= sum_weights
#            self.subsample_size = int(sum_weights)
#            return result
#
#        self.num += 1
#        # choose the rule to change (add/delete) at random
#        rule = self.rules_list[random.randrange(self.len_ruleset)]
#        # compute the likelihoods with/without the rule
#        new_logl = likelihood(self.deleted_rules ^ {rule})
##        print(new_logl, self.logl, new_logl-self.logl, self.temperature, (new_logl-self.logl) / self.temperature)
#        acc_prob = 1.0 if new_logl > self.logl else math.exp((new_logl-self.logl) / self.temperature)
#        # accept with probability exp(dh/T) (dh - likelihood change)
#        if acc_prob >= 1 or acc_prob >= random.random():
#            if rule in self.deleted_rules:
#                logfp.write('restoring: '+rule+' num_rules=' + str(self.len_ruleset-len(self.deleted_rules))+\
#                            ' subsample_size='+str(self.subsample_size)+' old_logl='+str(self.logl)+\
#                            ' new_logl='+str(new_logl)+' best_logl='+str(self.best_logl)+'\n')
#            else:
#                logfp.write('deleting: '+rule+' num_rules=' + str(self.len_ruleset-len(self.deleted_rules))+\
#                            ' subsample_size='+str(self.subsample_size)+' old_logl='+str(self.logl)+\
#                            ' new_logl='+str(new_logl)+' best_logl='+str(self.best_logl)+'\n')
#            self.deleted_rules ^= {rule}
#            self.logl = new_logl
#            if self.logl > self.best_logl:
#                self.best_deleted_rules = self.deleted_rules
#                self.best_logl = self.logl
#        # T decreases as G/log(iter_num) for some G
#        self.temperature = self.starting_temp / np.log2(self.num+1)
#    
#    def save_best_ruleset(self, filename):
#        rs = RuleSet()
#        for r in self.ruleset.values():
#            if not r.rule in self.best_deleted_rules:
#                rs[r.rule] = r
#        rs.save_to_file(filename)
#
#### AUXILIARY FUNCTIONS ###
#
#def optimize_rules(lexicon, edges, outfile):
#    deleted_rules = set()
#    iter_num = 0
#    while True:
#        iter_num += 1
#        print()
#        print('======== ITERATION %d =========' % iter_num)
#        print()
#        print('edges: %d' % len(edges))
#        print('rules: %d' % len(lexicon.ruleset))
#        lexicon.reset()
#        sampler = MCMCGraphSampler(lexicon, edges)
#        sampler.add_stat('int_without', RuleIntervalsWithoutStatistic(sampler))
#        print('Warming up the chain...')
#        pp = progress_printer(NUM_WARMUP_ITERATIONS)
#        while sampler.num < NUM_WARMUP_ITERATIONS:
#            num = sampler.num
#            sampler.next()
#            if sampler.num > num: next(pp)
#        sampler.reset()
#        print('Sampling graphs...')
#        sample = np.zeros(NUM_ITERATIONS)
#        pp = progress_printer(NUM_ITERATIONS)
#        while sampler.num < NUM_ITERATIONS:
#            num = sampler.num
#            sampler.next()
#            if sampler.num > num:
#                next(pp)
#                sample[sampler.num-1] = sampler.logl
#        sampler.update_stats()
#        print('Sampling rules...')
#        rule_sampler = MCMCRuleSetSampler(sample, sampler.stats['int_without'].intervals,\
#            sampler.lexicon.ruleset, ANNEALING_TEMPERATURE)
#        with open_to_write('logs/sampling-rules.log.'+str(iter_num)) as logfp:
#            pp = progress_printer(NUM_RULE_ITERATIONS)
#            while rule_sampler.num < NUM_RULE_ITERATIONS:
#                next(pp)
#                rule_sampler.next(logfp)
#        if rule_sampler.best_deleted_rules == deleted_rules:
#            rule_sampler.save_best_ruleset(outfile)
#            break
#        else:
#            deleted_rules = rule_sampler.best_deleted_rules
#            edges = list(filter(lambda e: e[2] not in deleted_rules, edges))
#            for rule in deleted_rules:
#                del sampler.lexicon.ruleset[rule]

def load_edges(filename):
    return list(read_tsv_file(filename, (str, str, str)))

def save_edges(sampler, filename):
    stats, stat_names = [], []
    for stat_name, stat in sorted(sampler.stats.items(), key = itemgetter(0)):
        if isinstance(stat, EdgeStatistic):
            stat_names.append(stat_name)
            stats.append(stat)
    with open_to_write(filename) as fp:
        write_line(fp, ('word_1', 'word_2', 'rule') + tuple(stat_names))
        for i, edge in enumerate(sampler.edges):
            write_line(fp, (str(edge.source), str(edge.target), str(edge.rule)) + tuple([stat.value(i) for stat in stats]))

def save_rules(sampler, filename):
    stats, stat_names = [], []
    for stat_name, stat in sorted(sampler.stats.items(), key = itemgetter(0)):
        if isinstance(stat, RuleStatistic):
            stat_names.append(stat_name)
            stats.append(stat)
    with open_to_write(filename) as fp:
        write_line(fp, ('rule', 'domsize') + tuple(stat_names))
        for rule in sampler.model.rule_features:
            write_line(fp, (str(rule), sampler.model.rule_features[rule][0].trials) +\
                           tuple([stat.value(rule) for stat in stats]))

def print_scalar_stats(sampler):
    stats, stat_names = [], []
    print()
    print()
    print('SIMULATION STATISTICS')
    print()
    spacing = max([len(stat_name)\
                   for stat_name, stat in sampler.stats.items() 
                       if isinstance(stat, ScalarStatistic)]) + 2
    for stat_name, stat in sorted(sampler.stats.items(), key = itemgetter(0)):
        if isinstance(stat, ScalarStatistic):
            print((' ' * (spacing-len(stat_name)))+stat_name, ':', stat.value())
    print()
    print()

def save_intervals(intervals, filename):
    with open_to_write(filename) as fp:
        for rule, ints in intervals.items():
            write_line(fp, (rule, len(ints), ' '.join([str(i) for i in ints])))

def mcmc_inference_preliminary(lexicon, model, edges):
    iter_num = 0
    while iter_num < settings.EM_MAX_ITERATIONS:
        iter_num += 1
        # init sampler
        sampler = MCMCGraphSampler(model, lexicon, edges)
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('graphs_without_1', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':/:lage']))))
        sampler.add_stat('graphs_without_2', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t']))))
        sampler.add_stat('graphs_without_3', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t', '{CAP}se:/:']))))
        sampler.add_stat('graphs_without_4', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t', '{CAP}se:/:', '{CAP}:/a:e']))))
        sampler.add_stat('graphs_without_5', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t', '{CAP}se:/:', '{CAP}:/a:e', ':/:ge']))))
        sampler.add_stat('graphs_without_6', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':/e:en']))))
        sampler.add_stat('graphs_without_7', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, [':{CAP}/en:ung']))))
        sampler.add_stat('graphs_without_8', GraphsWithoutRuleSetStatistic(sampler,\
            set(map(Rule.from_string, ['gül:zukünf/:']))))
        sampler.add_stat('exp_logl', ExpectedLogLikelihoodStatistic(sampler))
        sampler.add_stat('exp_contrib', RuleExpectedContributionStatistic(sampler))

        print('Warming up the sampler...')
        pp = progress_printer(settings.SAMPLING_WARMUP_ITERATIONS)
#        sampler.accept_all = True
        for i in pp:
            sampler.next()
#        sampler.accept_all = False

        # sample the graphs
        print('Sampling...')
        sampler.reset()
        pp = progress_printer(settings.SAMPLING_ITERATIONS)
        for i in pp:
            sampler.next()
        sampler.update_stats()

        print_scalar_stats(sampler)
        save_edges(sampler, 'edges_sample.txt.%d' % iter_num)
        save_rules(sampler, 'rule_stats.txt.%d' % iter_num)
        break

def mcmc_inference(lexicon, model, edges):
    iter_num = 0
    rule_sampler = MCMCAnnealingRuleSampler(model, lexicon, edges)
    while iter_num < settings.EM_MAX_ITERATIONS:
        iter_num += 1
        print('Iteration %d' % iter_num)
        print('num_rules = %d' % rule_sampler.model.num_rules())
        print('cost = %f' % rule_sampler.cost)
        rule_sampler.next()

