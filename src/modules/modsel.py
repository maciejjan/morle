from datastruct.graph import FullGraph
from datastruct.lexicon import Lexicon
from datastruct.rules import Rule
from models.marginal import MarginalModel
import algorithms.mcmc
import shared
import logging


def run() -> None:
    # load the lexicon
    logging.getLogger('main').info('Loading lexicon...')
    lexicon = Lexicon(shared.filenames['wordlist'])
#     logging.getLogger('main').info('Loading ruleset...')
#     ruleset = load_ruleset(shared.filenames['rules'])   # type: Dict[str, Rule]
#     logging.getLogger('main').info('Loading full graph...')

    # initialize a MarginalModel
    model = MarginalModel()
    model.fit_rootdist(lexicon.entries())
    model.load_rules_from_file(shared.filenames['rules'])
    model.fit_ruledist(model.iter_rules())

    # load the full graph
    full_graph = FullGraph(lexicon)
    full_graph.load_edges_from_file(shared.filenames['graph'])

    # inference
    logging.getLogger('main').info('Starting MCMC inference.')
    algorithms.mcmc.mcmc_inference(model, full_graph)

