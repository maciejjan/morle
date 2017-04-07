import algorithms.mcmc.inference
from datastruct.graph import FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
from models.marginal import MarginalModel
from utils.files import read_tsv_file
import shared
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

    # initialize a MarginalModel
    logging.getLogger('main').info('Initializing the model...')
    model = MarginalModel()
    model.fit_rootdist(lexicon.entries())
    model.fit_ruledist(rule for (rule, domsize) in rules)
    for rule, domsize in rules:
        model.add_rule(rule, domsize)

    # inference
    logging.getLogger('main').info('Starting MCMC inference.')
    algorithms.mcmc.inference.optimize_rule_set(full_graph, model)

