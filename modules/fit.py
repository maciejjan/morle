from datastruct.lexicon import *
from datastruct.rules import *
from models.point import PointModel
import algorithms.em
import shared
import logging

# model selection (with simulated annealing)
def prepare_model():
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
    model = PointModel(lexicon, None)
    model.fit_ruledist(set(rules.values()))
    for rule, domsize in rule_domsizes.items():
        model.add_rule(rules[rule], domsize)
#    model.save_to_file(model_filename)
    return model, lexicon, edges

# fit the model (with soft-EM algorithm)
def run():
    model, lexicon, edges = prepare_model()
    algorithms.em.softem(lexicon, model, edges)

