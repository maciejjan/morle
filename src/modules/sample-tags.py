import algorithms.mcmc.samplers
import algorithms.mcmc.statistics as stats
from datastruct.graph import EdgeSet, FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from models.suite import ModelSuite
from utils.files import file_exists, open_to_write, write_line
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

    # load the graph
    logging.getLogger('main').info('Loading graph edges...')
    edge_set = EdgeSet.load('possible-edges.txt', lexicon, rule_set)

    full_graph = FullGraph(lexicon, edge_set)

    model = ModelSuite.load()
    tagset = model.root_tag_model.tagset
    sampler = \
        algorithms.mcmc.samplers.MCMCImprovedTagSampler(\
            full_graph, model, tagset,
            warmup_iter=shared.config['sample-tags']\
                              .getint('warmup_iterations'),
            sampling_iter=shared.config['sample-tags']\
                                .getint('sampling_iterations'))
#             temperature_fun = lambda x: max(1.0, 10/math.log(x/10000+2.7)))
#     sampler.add_stat('edge_freq', stats.EdgeFrequencyStatistic(sampler))
#     sampler.add_stat('acc_rate', stats.AcceptanceRateStatistic(sampler))
    sampler.run_sampling()

    with open_to_write('tags.txt') as outfp:
        for w_id in range(len(lexicon)):
            tag_str = ' '.join([''.join(tag)+':'+str(sampler.tag_freq[w_id,t_id]) \
                               for t_id, tag in enumerate(tagset)])
            write_line(outfp, (lexicon[w_id], tag_str))
#     sampler.save_edge_stats(shared.filenames['sample-edge-stats'])
#     sampler.print_scalar_stats()

