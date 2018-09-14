from datastruct.graph import FullGraph, EdgeSet
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from models.suite import ModelSuite
from utils.files import file_exists
import shared

import logging


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

    for iter_num in range(shared.config['modsel'].getint('iterations'):
        sampler = MCMCGraphSampler(full_graph, model,
                shared.config['modsel'].getint('warmup_iterations'),
                shared.config['modsel'].getint('sampling_iterations'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()

        # compute the rule statistics

        # determine the rules to delete

        # delete the selected rules
    
    logging.getLogger('main').info('Saving the model...')
    model.save()

#     # load the lexicon
#     logging.getLogger('main').info('Loading lexicon...')
#     lexicon = Lexicon(shared.filenames['wordlist'])
# 
#     logging.getLogger('main').info('Loading rules...')
#     rule_set = RuleSet.load(shared.filenames['rules'])
# 
#     # load the full graph
#     logging.getLogger('main').info('Loading the graph...')
#     full_graph = FullGraph(lexicon)
#     full_graph.load_edges_from_file(shared.filenames['graph'])
# 
#     # initialize a MarginalModel
#     logging.getLogger('main').info('Initializing the model...')
#     model = ModelSuite(rule_set, lexicon=lexicon)
#     model.initialize(full_graph)
#     model.save()
# #     model.fit_rootdist(lexicon.entries())
# #     model.fit_ruledist(rule for (rule, domsize) in rules)
# #     for rule, domsize in rules:
# #         model.add_rule(rule, domsize)
# 
#     # inference
# #     logging.getLogger('main').info('Starting MCMC inference.')
# #     algorithms.mcmc.inference.optimize_rule_set(full_graph, model)
# 
