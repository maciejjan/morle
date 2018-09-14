from algorithms.mcmc.statistics import AcceptanceRateStatistic, \
    EdgeFrequencyStatistic, ExpectedCostStatistic
from algorithms.mcmc.samplers import MCMCGraphSampler
from datastruct.graph import FullGraph
# from models.point import PointModel
from models.suite import ModelSuite
import shared

import numpy as np
import logging


# def hardem(full_graph :FullGraph, model :PointModel) -> None:
#     iter_num = 0
#     model.recompute_root_costs(full_graph.nodes_iter())
#     while iter_num < shared.config['fit'].getint('iterations'):
#         iter_num += 1
#         logging.getLogger('main').info('Iteration %d' % iter_num)
#         model.recompute_edge_costs(full_graph.iter_edges())
# 
#         # expectation step
#         # TODO replace with optimal branching
#         branching = full_graph.optimal_branching(model)
# 
#         # maximization step
#         sample = list((edge, 
#                        (1.0 if branching.has_edge(*edge.key()) else 0.0))\
#                       for edge in full_graph.iter_edges())
#         num_edges = sum(1 for e, w in sample if w == 1.0)
#         model.fit_to_branching(branching)
#         logging.getLogger('main').info('num_edges = %f' % num_edges)
#         model.fit_to_sample(sample)
#         # TODO remove rules with zero frequency
#         model.save_rules_to_file(shared.filenames['rules-fit'])
# 
#         total_cost = model.roots_cost + model.rules_cost + model.edges_cost
#         logging.getLogger('main').info('cost = %f' % total_cost)
#         logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
#         logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
#         logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)


def softem(full_graph :FullGraph, model :ModelSuite) -> None:
    iter_num = 0
    # initialize the models
    model.initialize(full_graph)
#     model.root_model.fit(full_graph.lexicon, np.ones(len(full_graph.lexicon)))
#     model.fit(full_graph.lexicon, full_graph.edge_set,
#               np.ones(len(full_graph.lexicon)),
#               np.ones(len(full_graph.edge_set)))
    # EM iteration
    while iter_num < shared.config['fit'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)

        # expectation step
        sampler = MCMCGraphSampler(full_graph, model,
                warmup_iter=shared.config['fit'].getint('warmup_iterations'),
                sampling_iter=shared.config['fit'].getint('sampling_iterations'),
                depth_cost=shared.config['Models'].getfloat('depth_cost'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()

        # maximization step
        edge_weights = sampler.stats['edge_freq'].value()
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


