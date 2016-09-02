from datastruct.lexicon import *
from datastruct.rules import *
from models.marginal import MarginalModel
from utils.files import *
from algorithms.mcmc import *
import shared
import logging

# model selection (with simulated annealing)
# TODO remove code duplication (with 'modsel')
def prepare_model():
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    logging.getLogger('main').info('Loading rules...')
    rules, rule_domsizes = {}, {}
    rules_file = shared.filenames['rules-modsel']\
                 if file_exists(shared.filenames['rules-modsel'])\
                 else shared.filenames['rules']
    for rule, freq, domsize in read_tsv_file(rules_file,\
            (str, int, int)):
        rules[rule] = Rule.from_string(rule)
        rule_domsizes[rule] = domsize
    logging.getLogger('main').info('Loading edges...')
    edges = []
    for w1, w2, r in read_tsv_file(shared.filenames['graph']):
        if r in rules:
            edges.append(LexiconEdge(lexicon[w1], lexicon[w2], rules[r]))
    model = MarginalModel(lexicon, None)
    model.fit_ruledist(set(rules.values()))
    for rule, domsize in rule_domsizes.items():
        model.add_rule(rules[rule], domsize)
#    model.save_to_file(model_filename)
    return model, lexicon, edges

def run():
    model, lexicon, edges = prepare_model()
    logging.getLogger('main').info('Loaded %d rules.' % len(model.rule_features))
    sampler = MCMCGraphSamplerFactory.new(model, lexicon, edges,
            shared.config['sample'].getint('warmup_iterations'),
            shared.config['sample'].getint('sampling_iterations'))
    sampler.add_stat('cost', ExpectedCostStatistic(sampler))
    sampler.add_stat('acc_rate', AcceptanceRateStatistic(sampler))
    sampler.add_stat('edge_freq', EdgeFrequencyStatistic(sampler))
    sampler.add_stat('contrib', RuleExpectedContributionStatistic(sampler))
    sampler.run_sampling()
    sampler.summary()

