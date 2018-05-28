from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from utils.files import file_exists
import shared

import logging


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])

    # load the rules
    logging.getLogger('main').info('Loading rules...')
    rules_file = shared.filenames['rules-modsel']
    if not file_exists(rules_file):
        rules_file = shared.filenames['rules']
    rule_set = RuleSet.load(rules_file)
    # TODO compute the graph of possible edges
    # TODO save the graph
    pass

# from algorithms.mcmc.samplers import MCMCGraphSamplerFactory
# import algorithms.mcmc.statistics as stats
# from datastruct.graph import EdgeSet, FullGraph
# from datastruct.lexicon import Lexicon
# from datastruct.rules import Rule, RuleSet
# from models.suite import ModelSuite
# from utils.files import file_exists, read_tsv_file
# import algorithms.mcmc
# import shared
# import logging
# 
# 
# def run() -> None:
#     logging.getLogger('main').info('Loading lexicon...')
#     lexicon = Lexicon.load(shared.filenames['wordlist'])
# 
#     logging.getLogger('main').info('Loading rules...')
#     rules_file = shared.filenames['rules-modsel']
#     if not file_exists(rules_file):
#         rules_file = shared.filenames['rules']
#     rule_set = RuleSet.load(rules_file)
# 
#     edges_file = shared.filenames['graph-modsel']
#     if not file_exists(edges_file):
#         edges_file = shared.filenames['graph']
#     logging.getLogger('main').info('Loading the graph...')
#     edge_set = EdgeSet.load(edges_file, lexicon, rule_set)
#     full_graph = FullGraph(lexicon, edge_set)
# 
#     # initialize a ModelSuite
#     logging.getLogger('main').info('Loading the model...')
#     model = ModelSuite.load()
# 
#     # setup the sampler
#     logging.getLogger('main').info('Setting up the sampler...')
#     sampler = MCMCGraphSamplerFactory.new(full_graph, model,
#             shared.config['sample'].getint('warmup_iterations'),
#             shared.config['sample'].getint('sampling_iterations'),
#             shared.config['sample'].getint('iter_stat_interval'))
#     if shared.config['sample'].getboolean('stat_cost'):
#         sampler.add_stat('cost', stats.ExpectedCostStatistic(sampler))
#     if shared.config['sample'].getboolean('stat_acc_rate'):
#         sampler.add_stat('acc_rate', stats.AcceptanceRateStatistic(sampler))
#     if shared.config['sample'].getboolean('stat_iter_cost'):
#         sampler.add_stat('iter_cost', stats.CostAtIterationStatistic(sampler))
#     if shared.config['sample'].getboolean('stat_edge_freq'):
#         sampler.add_stat('edge_freq', stats.EdgeFrequencyStatistic(sampler))
#     if shared.config['sample'].getboolean('stat_undirected_edge_freq'):
#         sampler.add_stat('undirected_edge_freq', 
#                          stats.UndirectedEdgeFrequencyStatistic(sampler))
#     if shared.config['sample'].getboolean('stat_rule_freq'):
#         sampler.add_stat('freq', stats.RuleFrequencyStatistic(sampler))
#     if shared.config['sample'].getboolean('stat_rule_contrib'):
#         sampler.add_stat('contrib', 
#                          stats.RuleExpectedContributionStatistic(sampler))
# 
#     # run sampling and print results
#     logging.getLogger('main').info('Running sampling...')
#     sampler.run_sampling()
#     sampler.summary()
# 
