from algorithms.mcmc.samplers import MCMCGraphSamplerFactory
import algorithms.mcmc.statistics as stats
from datastruct.graph import FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
from models.marginal import MarginalModel
from utils.files import read_tsv_file
import algorithms.mcmc
import shared
import logging


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rules = [(Rule.from_string(rule_str), domsize) \
             for rule_str, domsize in \
                 read_tsv_file(shared.filenames['rules'], (str, int))]

    # load the full graph
    logging.getLogger('main').info('Loading the graph...')
    full_graph = FullGraph(lexicon)
    full_graph.load_edges_from_file(shared.filenames['graph'])

    # initialize a MarginalModel
    logging.getLogger('main').info('Initializing the model...')
    model = MarginalModel()
    model.fit_rootdist(lexicon.entries())
    model.fit_ruledist(rule for (rule, domsize) in rules)
    for rule, domsize in rules:
        model.add_rule(rule, domsize)

    # setup the sampler
    logging.getLogger('main').info('Setting up the sampler...')
    sampler = MCMCGraphSamplerFactory.new(full_graph, model,
            shared.config['sample'].getint('warmup_iterations'),
            shared.config['sample'].getint('sampling_iterations'),
            shared.config['sample'].getint('iter_stat_interval'))
    if shared.config['sample'].getboolean('stat_cost'):
        sampler.add_stat('cost', stats.ExpectedCostStatistic(sampler))
    if shared.config['sample'].getboolean('stat_acc_rate'):
        sampler.add_stat('acc_rate', stats.AcceptanceRateStatistic(sampler))
    if shared.config['sample'].getboolean('stat_iter_cost'):
        sampler.add_stat('iter_cost', stats.CostAtIterationStatistic(sampler))
    if shared.config['sample'].getboolean('stat_edge_freq'):
        sampler.add_stat('edge_freq', stats.EdgeFrequencyStatistic(sampler))
    if shared.config['sample'].getboolean('stat_undirected_edge_freq'):
        sampler.add_stat('undirected_edge_freq', 
                         stats.UndirectedEdgeFrequencyStatistic(sampler))
    if shared.config['sample'].getboolean('stat_rule_freq'):
        sampler.add_stat('freq', stats.RuleFrequencyStatistic(sampler))
    if shared.config['sample'].getboolean('stat_rule_contrib'):
        sampler.add_stat('contrib', 
                         stats.RuleExpectedContributionStatistic(sampler))
    logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
    logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
    logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)

    # run sampling and print results
    logging.getLogger('main').info('Running sampling...')
    sampler.run_sampling()
    sampler.summary()

    logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
    logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
    logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)

