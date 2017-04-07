from algorithms.mcmc.samplers import MCMCGraphSamplerFactory
from algorithms.mcmc.statistics import ExpectedCostStatistic, \
    AcceptanceRateStatistic, RuleExpectedContributionStatistic
from datastruct.graph import FullGraph
from datastruct.rules import Rule
from models.generic import Model
from models.marginal import MarginalModel
import shared

import logging
import random
from scipy.special import expit
from typing import Dict, Set, Tuple


class RuleSetProposalDistribution:
    def __init__(self, rule_scores :Dict[Rule, float], 
                 temperature :float) -> None:
        self.rule_prob = {}     # type: Dict[Rule, float]
        for rule, score in rule_scores.items():
#             rule_score = -rule_costs[rule] +\
#                 (rule_contrib[rule] if rule in rule_contrib else 0)
            self.rule_prob[rule] = expit(score * temperature)

    def propose(self) -> Set[Rule]:
        next_ruleset = set()        # type: Set[Rule]
        for rule, prob in self.rule_prob.items():
            if random.random() < prob:
                next_ruleset.add(rule)
        return next_ruleset

    def proposal_logprob(self, ruleset :Set[Rule]) -> float:
        return sum((np.log(prob) if rule in ruleset else np.log(1-prob)) \
                   for rule, prob in self.rule_prob.items())


class MCMCRuleSetOptimizer:
    def __init__(self, full_graph :FullGraph, model :Model,
                 warmup_iter :int = 0, sampling_iter: int = 0, 
                 alpha :float = 1, beta :float = 0.01) -> None:
        self.iter_num = 0
        self.model = model
        self.full_graph = full_graph
        self.current_ruleset = set(self.model.rule_features.keys())
        self.rule_domsize = {}      # type: Dict[Rule, int]
        self.warmup_iter = warmup_iter
        self.sampling_iter = sampling_iter
        self.alpha = alpha
        self.beta = beta
        self.update_temperature()
        for rule in self.current_ruleset:
            self.rule_domsize[rule] = \
                self.model.rule_features[rule][0].trials
        self.cost, self.proposal_dist = \
            self.evaluate_proposal(self.current_ruleset)

    def next(self) -> None:
        logging.getLogger('main').debug('temperature = %f' % self.temperature)
        next_ruleset = self.proposal_dist.propose()
#        self.print_proposal(next_ruleset)
        cost, next_proposal_dist = self.evaluate_proposal(next_ruleset)
        acc_prob = 1 if cost < self.cost else \
            math.exp((self.cost - cost) * self.temperature) *\
            math.exp(next_proposal_dist.proposal_logprob(self.current_ruleset) -\
                     self.proposal_dist.proposal_logprob(next_ruleset))
        logging.getLogger('main').debug('acc_prob = %f' % acc_prob)
        if random.random() < acc_prob:
            self.cost = cost
            self.proposal_dist = next_proposal_dist
            self.accept_ruleset(next_ruleset)
            logging.getLogger('main').debug('accepted')
        else:
            logging.getLogger('main').debug('rejected')
        self.iter_num += 1
        self.update_temperature()

    def evaluate_proposal(self, ruleset :Set[Rule]) \
                         -> Tuple[float, RuleSetProposalDistribution]:
#        self.model.reset()
        new_model = MarginalModel()
        new_model.rootdist = self.model.rootdist
        new_model.ruledist = self.model.ruledist
#        new_model.roots_cost = self.model.roots_cost
        for rule in ruleset:
            new_model.add_rule(rule, self.rule_domsize[rule])
#         self.lexicon.reset()
#         new_model.reset()
#         new_model.add_lexicon(self.lexicon)
#        print(new_model.roots_cost, new_model.rules_cost, new_model.edges_cost, new_model.cost())

#         graph_sampler = MCMCGraphSamplerFactory.new(new_model, self.lexicon,\
#             [edge for edge in self.edges if edge.rule in ruleset],\
#             self.warmup_iter, self.sampl_iter)
        graph_sampler = MCMCGraphSamplerFactory.new(
                            self.full_graph.restriction_to_ruleset(ruleset),
                            new_model, 
                            warmup_iter=self.warmup_iter,
                            sampling_iter=self.sampling_iter)
        graph_sampler.add_stat('cost', ExpectedCostStatistic(graph_sampler))
        graph_sampler.add_stat('acc_rate', AcceptanceRateStatistic(graph_sampler))
        graph_sampler.add_stat('contrib', RuleExpectedContributionStatistic(graph_sampler))
        graph_sampler.run_sampling()
        graph_sampler.log_scalar_stats()

        return graph_sampler.stats['cost'].val,\
            RuleSetProposalDistribution(
                graph_sampler.stats['contrib'].values_dict(), self.temperature)

    def accept_ruleset(self, new_ruleset :Set[Rule]) -> None:
        for rule in self.current_ruleset - new_ruleset:
            self.model.remove_rule(rule)
        for rule in new_ruleset - self.current_ruleset:
            self.model.add_rule(rule, self.rule_domsize[rule])
        self.current_ruleset = new_ruleset

    def print_proposal(self, new_ruleset :Set[Rule]) -> None:
        for rule in self.current_ruleset - new_ruleset:
            print('delete: %s' % str(rule))
        for rule in new_ruleset - self.current_ruleset:
            print('restore: %s' % str(rule))

    def update_temperature(self) -> None:
        self.temperature = (self.iter_num + self.alpha) * self.beta

    def save_rules(self, filename :str) -> None:
        self.model.save_rules_to_file(filename)

    def save_graph(self, filename :str) -> None:
        self.full_graph.restriction_to_ruleset(self.current_ruleset)\
            .save_to_file(filename)


#### AUXILIARY FUNCTIONS ###


def optimize_rule_set(full_graph :FullGraph, model :Model) -> None:
    # initialize the rule sampler
    warmup_iter = shared.config['modsel'].getint('warmup_iterations')
    sampling_iter = shared.config['modsel'].getint('sampling_iterations')
    alpha = shared.config['modsel'].getfloat('annealing_alpha')
    beta = shared.config['modsel'].getfloat('annealing_beta')
    rule_sampler = MCMCRuleSetOptimizer(
                       full_graph, model, warmup_iter=warmup_iter,
                       sampling_iter=sampling_iter, alpha=alpha, beta=beta)
    # main loop -- perfom the inference
    iter_num = 0
    while iter_num < shared.config['modsel'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)
        logging.getLogger('main').info(\
            'num_rules = %d' % rule_sampler.model.num_rules())
        logging.getLogger('main').info('cost = %f' % rule_sampler.cost)
        rule_sampler.next()
        rule_sampler.save_rules(shared.filenames['rules-modsel'])
        rule_sampler.save_graph(shared.filenames['graph-modsel'])

