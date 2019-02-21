#TODO deprecated module

import algorithms.clustering
from algorithms.mcmc.samplers import MCMCGraphSampler
from algorithms.mcmc.statistics import *
from datastruct.graph import FullGraph, EdgeSet
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule, RuleSet
from models.suite import ModelSuite
from utils.files import file_exists, open_to_write
import shared

import logging


def save_clusters(clusters, filename):
    with open_to_write(filename) as fp:
        for cluster in clusters:
            fp.write(', '.join([str(node) for node in cluster])+'\n')


def run() -> None:
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon.load(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rules_file = shared.filenames['rules-modsel']
    if not file_exists(rules_file):
        rules_file = shared.filenames['rules']
    rule_set = RuleSet.load(rules_file)

    edges_file = shared.filenames['graph-modsel']
    if not file_exists(edges_file):
        edges_file = shared.filenames['graph']
    logging.getLogger('main').info('Loading the graph...')
    edge_set = EdgeSet.load(edges_file, lexicon, rule_set)
    full_graph = FullGraph(lexicon, edge_set)

    logging.getLogger('main').info('Initializing the model...')
    model = ModelSuite(rule_set, lexicon = lexicon)
    model.root_model.fit(full_graph.lexicon, np.ones(len(full_graph.lexicon)))
    model.fit(full_graph.lexicon, full_graph.edge_set,
              np.ones(len(full_graph.lexicon)),
              np.ones(len(full_graph.edge_set)))
    
    iter_num = 0
    while iter_num < shared.config['fit-cluster'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)

        # expectation step
        sampler = MCMCGraphSampler(full_graph, model,
                shared.config['fit-cluster'].getint('warmup_iterations'),
                shared.config['fit-cluster'].getint('sampling_iterations'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()
        sampler.save_edge_stats(shared.filenames['sample-edge-stats'])

        # clustering step
        edge_weights = sampler.stats['exp_edge_freq'].value()
        clusters = \
            algorithms.clustering.chinese_whispers(
                full_graph, edge_weights,
                threshold=shared.config['fit-cluster']\
                                .getfloat('threshold'),
                root_weights=shared.config['fit-cluster']\
                                    .getboolean('root_weights'))
        clusters_idx = { word : i for i, words in enumerate(clusters) \
                                  for word in words }
        save_clusters(clusters, 'clusters.txt')
        for i, edge in enumerate(full_graph.edge_set):
            if clusters_idx[edge.source] != clusters_idx[edge.target]:
                edge_weights[i] = 0

        # maximization step
        edge_weights = sampler.stats['exp_edge_freq'].value()
        root_weights = np.ones(len(full_graph.lexicon))
        for idx in range(edge_weights.shape[0]):
            root_id = \
                full_graph.lexicon.get_id(full_graph.edge_set[idx].target)
            root_weights[root_id] -= edge_weights[idx]
        model.fit(sampler.lexicon, sampler.edge_set, 
                  root_weights, edge_weights)
        model.save()

        logging.getLogger('main').info('cost = %f' %\
                sampler.stats['exp_cost'].value())

