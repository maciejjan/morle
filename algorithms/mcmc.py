import logging
import math
import numpy as np
import random
import sys
import time
from operator import itemgetter
from scipy.special import beta, betaln, expit
from datastruct.lexicon import *
from datastruct.rules import *
from models.point import PointModel
from models.marginal import MarginalModel
from utils.files import *
from utils.printer import *
import shared

# Beta distribution parameters
#ALPHA = 1
#BETA = 1
#
#NUM_WARMUP_ITERATIONS = 1000000
#NUM_ITERATIONS = 10000000
#NUM_RULE_ITERATIONS = 10000
#ANNEALING_TEMPERATURE = 50

class ImpossibleMoveException(Exception):
    pass

### STATISTICS ###

class MCMCStatistic:
    def __init__(self, sampler):        raise NotImplementedError()
    def reset(self, sampler):            raise NotImplementedError()
    def update(self, sampler):            raise NotImplementedError()
    def edge_added(self, sampler):        raise NotImplementedError()
    def edge_removed(self, sampler):    raise NotImplementedError()
    def next_iter(self, sampler):        raise NotImplementedError()

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

class ExpectedCostStatistic(ScalarStatistic):
    def update(self, sampler):
        pass
    
    def edge_added(self, sampler, idx, edge):
        pass

    def edge_removed(self, sampler, idx, edge):
        pass
    
    def next_iter(self, sampler):
        self.val = (self.val * (sampler.num-1) + sampler.model.cost()) / sampler.num

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
    def __init__(self, model, lexicon, edges, warmup_iter, sampl_iter):
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.edges_hash = defaultdict(lambda: list())
        self.edges_idx = {}
        for idx, e in enumerate(edges):
            self.edges_idx[e] = idx
            self.edges_hash[(e.source, e.target)].append(e)
#        for idx, e in enumerate(edges):
#            self.edges_hash[(e.source, e.target)] = (idx, e)
        self.len_edges = len(edges)
        self.num = 0        # iteration number
        self.stats = {}
        self.warmup_iter = warmup_iter
        self.sampl_iter = sampl_iter
#        self.accept_all = False
    
    def add_stat(self, name, stat):
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def logl(self):
        return self.model.cost()

    def run_sampling(self):
        logging.getLogger('main').info('Warming up the sampler...')
        pp = progress_printer(self.warmup_iter)
        for i in pp:
            self.next()
        self.reset()
        pp = progress_printer(self.sampl_iter)
        logging.getLogger('main').info('Sampling...')
        for i in pp:
            self.next()
        self.update_stats()

    def next(self):
        # increase the number of iterations
        self.num += 1

        # select an edge randomly
        edge_idx = random.randrange(self.len_edges)
        edge = self.edges[edge_idx]

        # try the move determined by the selected edge
        try:
            edges_to_add, edges_to_remove, prop_prob_ratio =\
                self.determine_move_proposal(edge)
            acc_prob = self.compute_acc_prob(\
                edges_to_add, edges_to_remove, prop_prob_ratio)
            if acc_prob >= 1 or acc_prob >= random.random():
                self.accept_move(edges_to_add, edges_to_remove)
            for stat in self.stats.values():
                stat.next_iter(self)
        # if move impossible -- discard this iteration
        except ImpossibleMoveException:
            self.num -= 1

    def determine_move_proposal(self, edge):
        if edge in edge.source.edges:
            return self.propose_deleting_edge(edge)
        elif edge.source.has_ancestor(edge.target):
            return self.propose_flip(edge)
        elif edge.target.parent is not None:
            return self.propose_swapping_parent(edge)
        else:
            return self.propose_adding_edge(edge)

    def propose_adding_edge(self, edge):
        return [edge], [], 1

    def propose_deleting_edge(self, edge):
        return [], [edge], 1

    def propose_flip(self, edge):
        if random.random() < 0.5:
            return self.propose_flip_1(edge)
        else:
            return self.propose_flip_2(edge)

    def propose_flip_1(self, edge):
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.edges_hash[(node_3, node_1)]:
            raise ImpossibleMoveException()

        edge_3_1 = random.choice(self.edges_hash[(node_3, node_1)])
        edge_3_2 = self.find_edge_in_lexicon(node_3, node_2)
        edge_4_1 = self.find_edge_in_lexicon(node_4, node_1)

        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        if edge_4_1 is not None:
            edges_to_remove.append(edge_4_1)
        else: raise Exception('!')
        edges_to_add.append(edge_3_1)
        prop_prob_ratio = (1/len(self.edges_hash[(node_3, node_1)])) /\
                          (1/len(self.edges_hash[(node_3, node_2)]))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def propose_flip_2(self, edge):
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.edges_hash[(node_3, node_5)]:
            raise ImpossibleMoveException()

        edge_2_5 = self.find_edge_in_lexicon(node_2, node_5)
        edge_3_2 = self.find_edge_in_lexicon(node_3, node_2)
        edge_3_5 = random.choice(self.edges_hash[(node_3, node_5)])

        if edge_2_5 is not None:
            edges_to_remove.append(edge_2_5)
        elif node_2 != node_5: raise Exception('!')
        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        edges_to_add.append(edge_3_5)
        prop_prob_ratio = (1/len(self.edges_hash[(node_3, node_5)])) /\
                          (1/len(self.edges_hash[(node_3, node_2)]))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def nodes_for_flip(self, edge):
        node_1, node_2 = edge.source, edge.target
        node_3 = node_2.parent\
                              if node_2.parent is not None\
                              else None
        node_4 = node_1.parent
        node_5 = node_4
        if node_5 != node_2:
            while node_5.parent != node_2: 
                node_5 = node_5.parent
        return node_1, node_2, node_3, node_4, node_5

    def find_edge_in_lexicon(self, source, target):
        edges = [e for e in source.edges if e.target == target] 
        return edges[0] if edges else None

    def propose_swapping_parent(self, edge):
        return [edge], [e for e in edge.target.parent.edges\
                          if e.target == edge.target], 1

    def compute_acc_prob(self, edges_to_add, edges_to_remove, prop_prob_ratio):
        return math.exp(\
                -self.model.cost_of_change(edges_to_add, edges_to_remove)) *\
               prop_prob_ratio

    def accept_move(self, edges_to_add, edges_to_remove):
#            print('Accepted')
        # remove edges and update stats
        for e in edges_to_remove:
            idx = self.edges_idx[e]
            self.lexicon.remove_edge(e)
            self.model.apply_change([], [e])
            for stat in self.stats.values():
                stat.edge_removed(self, idx, e)
        # add edges and update stats
        for e in edges_to_add:
            idx = self.edges_idx[e]
            self.lexicon.add_edge(e)
            self.model.apply_change([e], [])
            for stat in self.stats.values():
                stat.edge_added(self, idx, e)
    
    def reset(self):
        self.num = 0
        for stat in self.stats.values():
            stat.reset(self)

    def update_stats(self):
        for stat in self.stats.values():
            stat.update(self)

    def print_scalar_stats(self):
        stats, stat_names = [], []
        print()
        print()
        print('SIMULATION STATISTICS')
        print()
        spacing = max([len(stat_name)\
                       for stat_name, stat in self.stats.items() 
                           if isinstance(stat, ScalarStatistic)]) + 2
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, ScalarStatistic):
                print((' ' * (spacing-len(stat_name)))+stat_name, ':', stat.value())
        print()
        print()

    def log_scalar_stats(self):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, ScalarStatistic):
                logging.getLogger('main').info('%s = %f' % (stat_name, stat.value()))

    def save_edge_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, EdgeStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('word_1', 'word_2', 'rule') + tuple(stat_names))
            for i, edge in enumerate(self.edges):
                write_line(fp, (str(edge.source), str(edge.target), str(edge.rule)) + tuple([stat.value(i) for stat in stats]))

    def save_rule_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, RuleStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('rule', 'domsize') + tuple(stat_names))
            for rule in self.model.rule_features:
                write_line(fp, (str(rule), self.model.rule_features[rule][0].trials) +\
                               tuple([stat.value(rule) for stat in stats]))
            
    def summary(self):
        self.print_scalar_stats()
        self.save_edge_stats(shared.filenames['sample-edge-stats'])
        self.save_rule_stats(shared.filenames['sample-rule-stats'])

# TODO constructor arguments should be the same for every type
#      (pass ensured edges through the lexicon parameter?)
# TODO init_lexicon() at creation
class MCMCSemiSupervisedGraphSampler(MCMCGraphSampler):
    def __init__(self, model, lexicon, edges, ensured_conn, warmup_iter, sampl_iter):
        MCMCGraphSampler.__init__(self, model, lexicon, edges, warmup_iter, sampl_iter)
        self.ensured_conn = ensured_conn

    def determine_move_proposal(self, edge):
        edges_to_add, edges_to_remove, prop_prob_ratio =\
            MCMCGraphSampler.determine_move_proposal(self, edge)
        removed_conn = set((e.source, e.target) for e in edges_to_remove) -\
                set((e.source, e.target) for e in edges_to_add)
        if removed_conn & self.ensured_conn:
            raise ImpossibleMoveException()
        else:
            return edges_to_add, edges_to_remove, prop_prob_ratio


class MCMCSupervisedGraphSampler(MCMCGraphSampler):
    def __init__(self, model, lexicon, edges, warmup_iter, sampl_iter):
        logging.getLogger('main').debug('Creating a supervised graph sampler.')
        MCMCGraphSampler.__init__(self, model, lexicon, edges, warmup_iter, sampl_iter)
        self.init_lexicon()

    def init_lexicon(self):
        edges_to_add = []
        for key, edges in self.edges_hash.items():
            edges_to_add.append(random.choice(edges))
        self.accept_move(edges_to_add, [])

    def determine_move_proposal(self, edge):
        if edge in edge.source.edges:
            edge_to_add = random.choice(self.edges_hash[(edge.source, edge.target)])
            if edge_to_add == edge:
                raise ImpossibleMoveException()
            return [edge_to_add], [edge], 1
        else:
            edge_to_remove = self.find_edge_in_lexicon(edge.source, edge.target)
            return [edge], [edge_to_remove], 1

    def run_sampling(self):
        self.reset()
        MCMCGraphSampler.run_sampling(self)


# TODO semi-supervised
class MCMCGraphSamplerFactory:
    def new(*args):
        if shared.config['General'].getboolean('supervised'):
            return MCMCSupervisedGraphSampler(*args)
        else:
            return MCMCGraphSampler(*args)


class RuleSetProposalDistribution:
    def __init__(self, rule_contrib, rule_costs, temperature):
        self.rule_prob = {}
        for rule in rule_costs:
            rule_score = -rule_costs[rule] +\
                (rule_contrib[rule] if rule in rule_contrib else 0)
            self.rule_prob[rule] =\
                expit(rule_score * temperature)

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
    def __init__(self, model, lexicon, edges, warmup_iter, sampl_iter):
        self.num = 0
        self.model = model
        self.lexicon = lexicon
        self.edges = edges
        self.full_ruleset = set(model.rule_features)
        self.ruleset = self.full_ruleset
        self.rule_domsize = {}
        self.rule_costs = {}
        self.warmup_iter = warmup_iter
        self.sampl_iter = sampl_iter
        self.update_temperature()
        for rule in self.full_ruleset:
            self.rule_domsize[rule] = self.model.rule_features[rule][0].trials
            self.rule_costs[rule] = self.model.rule_cost(rule, self.rule_domsize[rule])
        self.cost, self.proposal_dist = \
            self.evaluate_proposal(self.ruleset)

    def next(self):
        next_ruleset = self.proposal_dist.propose()
#        self.print_proposal(next_ruleset)
        cost, next_proposal_dist = self.evaluate_proposal(next_ruleset)
        acc_prob = 1 if cost < self.cost else \
            math.exp((self.cost - cost) * self.temperature) *\
            math.exp(next_proposal_dist.proposal_logprob(self.ruleset) -\
                     self.proposal_dist.proposal_logprob(next_ruleset))
        logging.getLogger('main').debug('acc_prob = %f' % acc_prob)
        if random.random() < acc_prob:
            self.cost = cost
            self.proposal_dist = next_proposal_dist
            self.accept_ruleset(next_ruleset)
            logging.getLogger('main').debug('accepted')
        else:
            logging.getLogger('main').debug('rejected')
        self.num += 1
        self.update_temperature()

    def evaluate_proposal(self, ruleset):
#        self.model.reset()
        new_model = MarginalModel(None, None)
        new_model.rootdist = self.model.rootdist
        new_model.ruledist = self.model.ruledist
#        new_model.roots_cost = self.model.roots_cost
        for rule in ruleset:
            new_model.add_rule(rule, self.rule_domsize[rule])
        self.lexicon.reset()
        new_model.reset()
        new_model.add_lexicon(self.lexicon)
#        print(new_model.roots_cost, new_model.rules_cost, new_model.edges_cost, new_model.cost())

        graph_sampler = MCMCGraphSamplerFactory.new(new_model, self.lexicon,\
            [edge for edge in self.edges if edge.rule in ruleset],\
            self.warmup_iter, self.sampl_iter)
        graph_sampler.add_stat('cost', ExpectedCostStatistic(graph_sampler))
        graph_sampler.add_stat('acc_rate', AcceptanceRateStatistic(graph_sampler))
        graph_sampler.add_stat('contrib', RuleExpectedContributionStatistic(graph_sampler))
        graph_sampler.run_sampling()
        graph_sampler.log_scalar_stats()

        return graph_sampler.stats['cost'].val,\
            RuleSetProposalDistribution(
                graph_sampler.stats['contrib'].values,
                self.rule_costs, self.temperature)

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

    def update_temperature(self):
        alpha = shared.config['modsel'].getfloat('annealing_alpha')
        beta = shared.config['modsel'].getfloat('annealing_beta')
        self.temperature = (self.num + alpha) / beta

    def save_rules(self):
        # save rules
        with open_to_write(shared.filenames['rules-modsel']) as outfp:
            for rule, freq, domsize in read_tsv_file(shared.filenames['rules']):
                if Rule.from_string(rule) in self.model.rule_features:
                    write_line(outfp, (rule, freq, domsize))
        # save graph (TODO rename the function?)
        with open_to_write(shared.filenames['graph-modsel']) as outfp:
            for w1, w2, rule in read_tsv_file(shared.filenames['graph']):
                if Rule.from_string(rule) in self.model.rule_features:
                    write_line(outfp, (w1, w2, rule))


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


def save_intervals(intervals, filename):
    with open_to_write(filename) as fp:
        for rule, ints in intervals.items():
            write_line(fp, (rule, len(ints), ' '.join([str(i) for i in ints])))

#def mcmc_inference_preliminary(lexicon, model, edges):
#    iter_num = 0
#    while iter_num < settings.EM_MAX_ITERATIONS:
#        iter_num += 1
#        # init sampler
#        sampler = MCMCGraphSampler(model, lexicon, edges)
#        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
#        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
#        sampler.add_stat('graphs_without_1', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':/:lage']))))
#        sampler.add_stat('graphs_without_2', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t']))))
#        sampler.add_stat('graphs_without_3', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t', '{CAP}se:/:']))))
#        sampler.add_stat('graphs_without_4', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t', '{CAP}se:/:', '{CAP}:/a:e']))))
#        sampler.add_stat('graphs_without_5', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':/:lage', '{CAP}:/a:ü/:t', '{CAP}se:/:', '{CAP}:/a:e', ':/:ge']))))
#        sampler.add_stat('graphs_without_6', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':/e:en']))))
#        sampler.add_stat('graphs_without_7', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, [':{CAP}/en:ung']))))
#        sampler.add_stat('graphs_without_8', GraphsWithoutRuleSetStatistic(sampler,\
#            set(map(Rule.from_string, ['gül:zukünf/:']))))
#        sampler.add_stat('exp_logl', ExpectedLogLikelihoodStatistic(sampler))
#        sampler.add_stat('exp_contrib', RuleExpectedContributionStatistic(sampler))
#
#        print('Warming up the sampler...')
#        pp = progress_printer(settings.SAMPLING_WARMUP_ITERATIONS)
##        sampler.accept_all = True
#        for i in pp:
#            sampler.next()
##        sampler.accept_all = False
#
#        # sample the graphs
#        print('Sampling...')
#        sampler.reset()
#        pp = progress_printer(settings.SAMPLING_ITERATIONS)
#        for i in pp:
#            sampler.next()
#        sampler.update_stats()
#
#        print_scalar_stats(sampler)
#        save_edges(sampler, 'edges_sample.txt.%d' % iter_num)
#        save_rules(sampler, 'rule_stats.txt.%d' % iter_num)
#        break

def mcmc_inference(lexicon, model, edges):
    iter_num = 0
    rule_sampler = MCMCAnnealingRuleSampler(model, lexicon, edges,
            shared.config['modsel'].getint('warmup_iterations'),
            shared.config['modsel'].getint('sampling_iterations'))
    while iter_num < shared.config['modsel'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)
        logging.getLogger('main').info(\
            'num_rules = %d' % rule_sampler.model.num_rules())
        logging.getLogger('main').info('cost = %f' % rule_sampler.cost)
        rule_sampler.next()
        rule_sampler.save_rules()

