from datastruct.graphs import FullGraph
from models.neural import NeuralModel

# TODO
# - soft EM with a keras model
def softem(full_graph :FullGraph, model :NeuralModel) -> None:
    iter_num = 0
    model.recompute_root_costs(full_graph.nodes_iter())
    while iter_num < shared.config['fit'].getint('iterations'):
        iter_num += 1
        logging.getLogger('main').info('Iteration %d' % iter_num)
        model.recompute_edge_costs(full_graph.iter_edges())

        # expectation step
        sampler = MCMCGraphSampler(full_graph, model,
                shared.config['fit'].getint('warmup_iterations'),
                shared.config['fit'].getint('sampling_iterations'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()

        # maximization step
        sample = list((edge, sampler.stats['exp_edge_freq'].value(edge))\
                      for edge in sampler.edge_index)
        model.fit_to_sample(sample)
#         model.save_rules_to_file(shared.filenames['rules-fit'])
        # TODO save edge stats

        logging.getLogger('main').info('acc_rate = %f' %\
                sampler.stats['acc_rate'].value())
        logging.getLogger('main').info('cost = %f' %\
                sampler.stats['exp_cost'].value())
#         logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
#         logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
#         logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)


def run() -> None:
    pass

