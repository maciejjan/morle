import algorithms.mcmc.samplers
import algorithms.mcmc.statistics as stats
from datastruct.graph import EdgeSet, FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import RuleSet
from models.suite import ModelSuite
from utils.files import file_exists, open_to_write, write_line
import shared

import logging
import numpy as np
from operator import itemgetter


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

    # save baseline
    with open_to_write('tags-baseline.txt') as outfp:
        probs = model.root_tag_model.predict_tags(lexicon)
        for w_id in range(len(lexicon)):
            tag_probs = sorted([(tag, probs[w_id,t_id]) \
                                for t_id, tag in enumerate(tagset)],
                               reverse=True, key=itemgetter(1))
            tag_str = ' '.join([''.join(tag)+':'+str(prob) \
                               for (tag, prob) in tag_probs])
            write_line(outfp, (lexicon[w_id], tag_str))

    # prepare the sampler + run sampling
    sampler = \
        algorithms.mcmc.samplers.MCMCTagSampler(\
            full_graph, model, tagset,
            warmup_iter=shared.config['sample-tags']\
                              .getint('warmup_iterations'),
            sampling_iter=shared.config['sample-tags']\
                                .getint('sampling_iterations'))
    sampler.add_stat('edge_freq', stats.EdgeFrequencyStatistic(sampler))
    sampler.add_stat('acc_rate', stats.AcceptanceRateStatistic(sampler))
    sampler.run_sampling()

    # save the sampled tags and other statistics
    with open_to_write('tags.txt') as outfp:
        num_zero_probs = 0
        for w_id in range(len(lexicon)):
#             tag_probs = [(tag, sampler.tag_freq[w_id, t_id]) \
#                          for t_id, tag in enumerate(tagset)]
            probs = sampler.tag_freq[w_id,:]
            if np.sum(probs) <= 0:
                probs = model.root_tag_model.predict_tags([lexicon[w_id]])[0,:]
                num_zero_probs += 1
            tag_probs = sorted([(tag, probs[t_id]) \
                                for t_id, tag in enumerate(tagset)],
                               reverse=True, key=itemgetter(1))
            tag_str = ' '.join([''.join(tag)+':'+str(prob) \
                               for (tag, prob) in tag_probs])
            write_line(outfp, (lexicon[w_id], tag_str))
        logging.getLogger('main').info(\
            '{} words with zero probs ({} %).'\
            .format(num_zero_probs, num_zero_probs*100/len(lexicon)))
    sampler.save_edge_stats(shared.filenames['sample-edge-stats'])
    sampler.print_scalar_stats()


