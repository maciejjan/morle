from collections import defaultdict
import algorithms.branching
from algorithms.mcmc import *
#import algorithms.ngrams
#import algorithms.optrules
#import algorithms.align
#from algorithms.optrules import extract_all_rules
#from datastruct.lexicon import *
#from datastruct.rules import *
#from utils.files import *
from utils.printer import *
#import shared
#import re
#
#import math

def hardem(lexicon, model, edges):
    old_cost = lexicon.cost + model.cost
    old_num_rules = model.num_rules()
    iter_num = 0
    print('num_rules = %d' % old_num_rules)
    print('cost = %f' % old_cost)

    # main EM loop
#    while True:
    while iter_num < settings.EM_MAX_ITERATIONS:
        iter_num += 1
        print()
        print('=== Iteration %d ===' % iter_num)
        print('Resetting lexicon...')
        lexicon.reset(model)
#        lexicon.recompute_cost(model)
        # compute maximum branching
        print('Computing maximum branching...')
        vertices = list(lexicon.iter_nodes())
        edges = list(e for e in edges if model.has_rule(e.rule))
        branching = algorithms.branching.branching(vertices, edges)
        for e in branching:
            lexicon.add_edge(e)
        print('intermediate lexicon cost: %f' % lexicon.cost)
        # fit the model to the optimum lexicon
        print('Fitting the model...')
        model.fit_to_lexicon(lexicon)
        # recompute edge costs
        for e in edges:
            if model.has_rule(e.rule):
                e.cost = model.edge_cost(e)
            elif lexicon.has_edge(e):
                lexicon.remove_edge(e)
        lexicon.recompute_cost(model)
        # compute iteration statistics and interruption criterion
        num_rules = model.num_rules()
        cost = lexicon.cost + model.cost
        print('num_rules = %d' % num_rules)
        print('lexicon cost = %f' % lexicon.cost)
        print('model cost = %f' % model.cost)
        print('total cost = %f' % cost)
#        if 0 <= old_cost-cost <= 1 and num_rules == old_num_rules:
#            break
#        break # TODO test
        old_cost = cost
        old_num_rules = num_rules

def softem(lexicon, model, edges):
    iter_num = 0
    model.recompute_root_costs(lexicon.iter_nodes())
    while iter_num < shared.config['fit'].getint('iterations'):
        lexicon.reset()
        model.reset()
        model.recompute_edge_costs(edges)

        sampler = MCMCGraphSampler(model, lexicon, edges,
                shared.config['fit'].getint('warmup_iterations'),
                shared.config['fit'].getint('sampling_iterations'))
        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
        sampler.add_stat('exp_cost', ExpectedCostStatistic(sampler))
        sampler.run_sampling()

        sample = list((edge, sampler.stats['exp_edge_freq'].value(i))\
                      for i, edge in enumerate(sampler.edges))
        model.fit_to_sample(sample)
        model.save_rules(shared.filenames['rules-fit'])

        logging.getLogger('main').info('cost = %f' %\
                sampler.stats['exp_cost'].value())
        logging.getLogger('main').debug('roots_cost = %f' % model.roots_cost)
        logging.getLogger('main').debug('rules_cost = %f' % model.rules_cost)
        logging.getLogger('main').debug('edges_cost = %f' % model.edges_cost)

        # TODO show some debug info

#    old_cost = lexicon.cost + model.cost
#    old_num_rules = model.num_rules()
#    iter_num = 0
#    print('num_rules = %d' % old_num_rules)
#    print('cost = %f' % old_cost)
#
#    # main EM loop
##    while True:
#    while iter_num < settings.EM_MAX_ITERATIONS:
#        iter_num += 1
#        print()
#        print('=== Iteration %d ===' % iter_num)
#
#        # init sampler
#        lexicon.reset(model)
#        sampler = MCMCGraphSampler(model, lexicon, edges)
#        sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
#        sampler.add_stat('exp_edge_freq', EdgeFrequencyStatistic(sampler))
#        sampler.add_stat('exp_logl', ExpectedLogLikelihoodStatistic(sampler))
#        sampler.run_sampling()
##        print('Warming up the sampler...')
##        pp = progress_printer(settings.SAMPLING_WARMUP_ITERATIONS)
##        for i in pp:
##            sampler.next()
##
##        # sample the graphs
###        sample_stats = []
##        print('Sampling...')
##        sampler.reset()
##        pp = progress_printer(settings.SAMPLING_ITERATIONS)
##        for i in pp:
##            sampler.next()
###            sample_stats.append((sampler.lexicon.cost, sampler.stats['acc_rate'].value()))
##        sampler.update_stats()
#
#        # reformat the sampling results
#        lexicon_cost = sampler.stats['exp_logl'].value()
#        print('acceptance rate = %f' % sampler.stats['acc_rate'].value())
#        print('lexicon cost = %f' % lexicon_cost)
#        # TODO print the total cost here?
#        edges_by_rule = defaultdict(lambda: list())
#        for i, edge in enumerate(sampler.edges):
#            edges_by_rule[edge.rule].append(\
#                (edge, sampler.stats['exp_edge_freq'].value(i)))
#
##        cost = 0.0
##        for i, edge in enumerate(edges):
##            cost += edge.cost * sampler.stats['exp_edge_freq'].value(i)
##        cost += model.null_cost()
##        print('sample cost before fitting: %f' % cost)
#
#        # write data for reference
##        null_cost = model.null_cost()
##        if iter_num == 1:
##            with open_to_write('costs_sample.txt.%d' % iter_num) as fp:
##                for cost, acc in sample_stats:
##                    write_line(fp, (cost, acc))
##        if iter_num % 5 == 0:
##            with open_to_write('edges_sample.txt.%d' % iter_num) as fp:
##                for rule, edges_list in edges_by_rule.items():
##                    for edge, weight in edges_list:
##                        write_line(fp, (edge.source.key, edge.target.key, str(edge.rule), str(weight)))
##            model.save_rule_stats('rule_stats.txt.%d' % iter_num)
#
#        # fit the model
#        print('null cost before fitting = %f' % model.null_cost())
#        model.fit_to_sample(edges_by_rule)
#        print('null cost after fitting = %f' % model.null_cost())
#        print('model cost = %f' % model.cost)
#        num_rules = model.num_rules()
#        cost = lexicon_cost + model.cost
#        print('num_rules = %d' % num_rules)
#        print('total cost = %f' % cost)
#
#        # recompute the costs
#        edges = [e for e in edges if model.has_rule(e.rule)]
#        for e in edges:
#            e.cost = model.edge_cost(e)
##        lexicon.recompute_cost(model)
#
#        if iter_num % 5 == 0:
#            print('Saving edges...')
#            with open_to_write('edges.txt.%d' % iter_num) as fp:
#                for e in edges:
#                    write_line(fp, tuple(map(str, (e.source, e.target, e.rule, e.cost, e.target.cost, e.target.cost-e.cost))))
#
##        cost = 0.0
##        for i, edge in enumerate(edges):
##            cost += edge.cost * sampler.stats['exp_edge_freq'].value(i)
##        cost += model.null_cost()
##        print('sample cost after fitting: %f' % cost)
#
#        # interruption criterion
##        if 0 <= old_cost-cost <= 1 and num_rules == old_num_rules:
##        if (old_cost-cost <= 1 and num_rules == old_num_rules) or\
##                iter_num >= settings.EM_MAX_ITERATIONS:
##            break
#        old_cost = cost
#        old_num_rules = num_rules

def load_training_file_with_freq(filename):
    ruleset = RuleSet()
    rootdist = algorithms.ngrams.NGramModel(settings.ROOTDIST_N)
    rootdist.train_from_file(filename)
    lexicon = Lexicon(rootdist=rootdist, ruleset=ruleset)

    for word, freq in read_tsv_file(filename, (str, int)):
        lexicon[word] = LexiconNode(word, freq, rootdist.word_prob(word))
        lexicon.roots.add(word)
    for word_2, freq, word_1 in read_tsv_file(filename, (str, int, str),\
            print_progress=True, print_msg='Building lexicon from training data...'):
        if word_1 != u'-' and word_1 != word_2 and word_2 in lexicon.roots:
            if word_1 not in lexicon:
                lexicon.add_word(word_1, 1, rootdist.word_prob(word_1))
            rule = algorithms.align.align(word_1, word_2).to_string()
            if rule not in ruleset:
                ruleset[rule] = RuleData(rule, 1.0, 1, 0)
            lexicon.draw_edge(word_1, word_2, rule)
    return lexicon

def load_training_file_without_freq(filename):
    ruleset = RuleSet()
    rootdist = algorithms.ngrams.NGramModel(settings.ROOTDIST_N)
    rootdist.train([(word, 1) for (word,) in read_tsv_file(filename, (str,))])
    lexicon = Lexicon(rootdist=rootdist, ruleset=ruleset)

    for (word,) in read_tsv_file(filename, (str,)):
        lexicon[word] = LexiconNode(word, 1, rootdist.word_prob(word))
        lexicon.roots.add(word)
    for word_2, word_1 in read_tsv_file(filename, (str, str),\
            print_progress=True, print_msg='Building lexicon from training data...'):
        if word_1 != u'-' and word_1 != word_2 and word_2 in lexicon.roots:
            if word_1 not in lexicon:
                lexicon.add_word(word_1, 1, rootdist.word_prob(word_1))
            rule = algorithms.align.align(word_1, word_2).to_string()
            if rule not in ruleset:
                ruleset[rule] = RuleData(rule, 1.0, 1, 0)
            lexicon.draw_edge(word_1, word_2, rule)
    return lexicon

#TODO supervised
def load_training_file(filename):
    if settings.USE_WORD_FREQ:
        return load_training_file_with_freq(filename)
    else:
        return load_training_file_without_freq(filename)
    
