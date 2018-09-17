from algorithms.mcmc.samplers import MCMCGraphSampler
from algorithms.mcmc.statistics import \
    AcceptanceRateStatistic, EdgeFrequencyStatistic, ExpectedCostStatistic
from datastruct.graph import FullGraph, EdgeSet
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from models.suite import ModelSuite
from utils.files import file_exists
import shared

import logging
import numpy as np


def run() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rule_set = RuleSet.load(shared.filenames['rules'])

    logging.getLogger('main').info('Loading the graph...')
    edge_set = EdgeSet.load(shared.filenames['graph'], lexicon, rule_set)
    full_graph = FullGraph(lexicon, edge_set)

    logging.getLogger('main').info('Initializing the model...')
    model = ModelSuite(rule_set, lexicon = lexicon)
    model.initialize(full_graph)
    deleted_rules = set()

    for iter_num in range(shared.config['modsel'].getint('iterations')):
        sampler = MCMCGraphSampler(full_graph, model,
                shared.config['modsel'].getint('warmup_iterations'),
                shared.config['modsel'].getint('sampling_iterations'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()

        # compute the rule statistics
        freq, contrib = sampler.compute_rule_stats()

        # determine the rules to delete 
        deleted_rules |= set(np.where(contrib < 0)[0])
        logging.getLogger('main').info(\
            '{} rules deleted.'.format(len(deleted_rules)))

        # delete the edges with selected rules from the graph
        edges_to_delete = []
        for edge in full_graph.edges_iter():
            if model.rule_set.get_id(edge.rule) in deleted_rules:
                edges_to_delete.append(edge)
        full_graph.remove_edges(edges_to_delete)

        # delete the selected rules -- TODO implementation
#         model.delete_rules(rules_to_delete)

    logging.getLogger('main').info('Saving the graph...')
    full_graph.edge_set.save(shared.filenames['graph-modsel'])

    # remove the deleted rules from the rule set and save it
    logging.getLogger('main').info('Saving the rule set...')
    new_rule_set = RuleSet()
    for i, rule in enumerate(rule_set):
        if i not in deleted_rules:
            new_rule_set.add(rule, rule_set.get_domsize(rule))
    new_rule_set.save(shared.filenames['rules-modsel'])

