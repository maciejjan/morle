from datastruct.lexicon import *
from datastruct.rules import *
from models.marginal import MarginalModel
from utils.files import *
import algorithms.mcmc
import shared
import logging

# model selection (with simulated annealing)
def prepare_marginal_model():
    lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
    logging.getLogger('main').info('Loading rules...')
    rules, rule_domsizes = {}, {}
    for rule, freq, domsize in read_tsv_file(shared.filenames['rules'],\
            (str, int, int)):
        rules[rule] = Rule.from_string(rule)
        rule_domsizes[rule] = domsize
    logging.getLogger('main').info('Loading edges...')
    edges = []
    for w1, w2, r in read_tsv_file(shared.filenames['graph']):
        edges.append(LexiconEdge(lexicon[w1], lexicon[w2], rules[r]))
    model = MarginalModel(lexicon, None)
    model.fit_ruledist(set(rules.values()))
    for rule, domsize in rule_domsizes.items():
        model.add_rule(rules[rule], domsize)
#    model.save_to_file(model_filename)
    return model, lexicon, edges

def run():
    model, lexicon, edges = prepare_marginal_model()
    algorithms.mcmc.mcmc_inference(lexicon, model, edges)

def cleanup():
    remove_file_if_exists(shared.filenames['graph-modsel'])
    remove_file_if_exists(shared.filenames['rules-modsel'])

