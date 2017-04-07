import algorithms.em
from datastruct.graph import FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
from models.point import PointModel
from utils.files import read_tsv_file
import shared

from collections import defaultdict
import logging


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(shared.filenames['wordlist'])

    logging.getLogger('main').info('Loading rules...')
    rules = [(Rule.from_string(rule_str), domsize) \
             for rule_str, domsize in \
                 read_tsv_file(shared.filenames['rules'], (str, int))]

    # load the full graph
    logging.getLogger('main').info('Loading the graph...')
    full_graph = FullGraph(lexicon)
    full_graph.load_edges_from_file(shared.filenames['graph'])

    # count rule frequencies in the full graph
    rule_freq = defaultdict(lambda: 0)
    for edge in full_graph.iter_edges():
        rule_freq[edge.rule] += 1

    # initialize a PointModel
    logging.getLogger('main').info('Initializing the model...')
    model = PointModel()
    model.fit_rootdist(lexicon.entries())
    model.fit_ruledist(rule for (rule, domsize) in rules)
    for rule, domsize in rules:
        model.add_rule(rule, domsize, freq=rule_freq[rule])

    algorithms.em.softem(full_graph, model)

# def prepare_model():
#     lexicon = Lexicon.init_from_wordlist(shared.filenames['wordlist'])
#     logging.getLogger('main').info('Loading rules...')
#     rules, rule_domsizes = {}, {}
#     for rule, freq, domsize in read_tsv_file(shared.filenames['rules-modsel'],\
#             (str, int, int)):
#         rules[rule] = Rule.from_string(rule)
#         rule_domsizes[rule] = domsize
#     logging.getLogger('main').info('Loading edges...')
#     edges = []
#     for w1, w2, r in read_tsv_file(shared.filenames['graph-modsel']):
#         if r in rules:
#             edges.append(LexiconEdge(lexicon[w1], lexicon[w2], rules[r]))
#     model = PointModel(lexicon, None)
#     model.fit_ruledist(set(rules.values()))
#     for rule, domsize in rule_domsizes.items():
#         model.add_rule(rules[rule], domsize)
# #    model.save_to_file(model_filename)
# #     save_rootgen_transducer(model)
#     return model, lexicon, edges

# fit the model (with soft-EM algorithm)
# def run():
#     model, lexicon, edges = prepare_model()
#     algorithms.em.softem(lexicon, model, edges)
# 
# def cleanup():
#     remove_file_if_exists(shared.filenames['rules-fit'])

