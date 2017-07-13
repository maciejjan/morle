from algorithms.mcmc.statistics import AcceptanceRateStatistic, \
    EdgeFrequencyStatistic, ExpectedCostStatistic
from algorithms.mcmc.samplers import MCMCGraphSampler
from datastruct.graph import FullGraph
from models.point import PointModel
from models.neural import ModelSuite
import shared

import logging


def hardem(full_graph :FullGraph, model :PointModel) -> None:
    iter_num = 0
    model.recompute_root_costs(full_graph.nodes_iter())
    while iter_num < shared.config['fit'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)
        model.recompute_edge_costs(full_graph.iter_edges())

        # expectation step
        # TODO replace with optimal branching
        branching = full_graph.optimal_branching(model)

        # maximization step
        sample = list((edge, 
                       (1.0 if branching.has_edge(*edge.key()) else 0.0))\
                      for edge in full_graph.iter_edges())
        num_edges = sum(1 for e, w in sample if w == 1.0)
        model.fit_to_branching(branching)
        logging.getLogger('main').info('num_edges = %f' % num_edges)
        model.fit_to_sample(sample)
        # TODO remove rules with zero frequency
        model.save_rules_to_file(shared.filenames['rules-fit'])

        total_cost = model.roots_cost + model.rules_cost + model.edges_cost
        logging.getLogger('main').info('cost = %f' % total_cost)
        logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
        logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
        logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)


def softem(full_graph :FullGraph, model :ModelSuite) -> None:
    iter_num = 0
#     model.recompute_root_costs(full_graph.nodes_iter())
#     model.recompute_costs()
    while iter_num < shared.config['fit'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)
        model.recompute_costs()

        # expectation step
        sampler = MCMCGraphSampler(full_graph, model,
                shared.config['fit'].getint('warmup_iterations'),
                shared.config['fit'].getint('sampling_iterations'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()

        # maximization step
#         sample = list((edge, sampler.stats['exp_edge_freq'].value(edge))\
#                       for edge in sampler.edge_index)
        model.fit_to_sample(sampler.stats['exp_edge_freq'].value())
#         model.save_rules_to_file(shared.filenames['rules-fit'])
        model.save()

        logging.getLogger('main').info('cost = %f' %\
                sampler.stats['exp_cost'].value())
#         logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
#         logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
#         logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)


