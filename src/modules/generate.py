from algorithms.analyzer import Analyzer
import algorithms.fst
from datastruct.lexicon import *
from datastruct.graph import EdgeSet, FullGraph
from datastruct.rules import *
from models.suite import ModelSuite
from utils.files import *
import shared

from operator import itemgetter
import hfst
import logging
import math
from scipy.stats import norm


# TODO
# - generate from all rules, regardless of the cost
# - additionally: word frequency test
#   - fit the frequency model using a sampler
#   - (result: a mean and sdev for each rule)
#   - add the cost of a frequency below 1 to the cost of the word
# - sum the probabilities of a word over all possible derivations!
#   (pairs: source_word, rule)
# - result: pairs (word, cost)

# parameters: min_freq, max_freq -- generate the words, for which the predicted
# frequency falls into this interval
# evaluate on a real corpus, e.g. with frequencies between 1 and 5

# TODO 2018-10-15
# - remove the ad-hoc frequency model implemented here and load the 'real'
#   frequency model instead
# - test running the module without the frequency test
# - config parameters: min_freq, max_freq, max_cost
# - evaluate the tagged and untagged variant
# - additionally: test the generated words with a (separately trained?) root
#     distribution (e.g. trigram or ALERGIA without smoothing)
#     -- if root costs are very low, do not include

# def fit_frequency_model(sample_edge_stats_filename, lexicon, rule_set):
#     means, sdevs = np.zeros(len(rule_set)), np.zeros(len(rule_set))
#     values, weights = {}, {}
#     for w1, w2, rule, freq_str in read_tsv_file(sample_edge_stats_filename):
#         try:
#             freq = float(freq_str)
#             if freq <= 0:
#                 continue
#             if rule not in values:
#                 values[rule] = []
#                 weights[rule] = []
#             values[rule].append(freq * (lexicon[w2].logfreq - lexicon[w1].logfreq))
#             weights[rule].append(freq)
#         except ValueError:
#             pass
#     for rule in values:
#         r_id = rule_set.get_id(Rule.from_string(rule))
#         a_values = np.array(values[rule])
#         a_weights = np.array(weights[rule])
#         means[r_id] = np.average(a_values, weights=a_weights)
#         sdevs[r_id] = np.sqrt(np.average((a_values-means[r_id])**2, weights=a_weights))
#     return means, sdevs


def run() -> None:
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    analyzer_file = 'analyzer.fsm'
    analyzer = None
    if file_exists(analyzer_file):
        analyzer = Analyzer.load(analyzer_file, lexicon, model)
    else:
        analyzer = Analyzer(lexicon, model)
        analyzer.save(analyzer_file)
    tr_file = 'wordgen.fst'
    if not file_exists(tr_file):
        new_words_acceptor = hfst.HfstTransducer(analyzer.fst)
        new_words_acceptor.convert(hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        new_words_acceptor.input_project()
        new_words_acceptor.minimize()
        new_words_acceptor.subtract(lexicon.to_fst())
        new_words_acceptor.minimize()
        algorithms.fst.save_transducer(new_words_acceptor, tr_file)

    logging.getLogger('main').info('Fitting the frequency model...')
    means, sdevs = fit_frequency_model('sample-edge-stats.txt',
                                       lexicon, model.rule_set)

    logging.getLogger('main').info('Generating...')
    cmd = ['hfst-fst2strings', full_path(tr_file)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, 
                         universal_newlines=True, bufsize=1)
    with open_to_write(shared.filenames['wordgen']) as fp:
        while True:
            line = p.stdout.readline().strip()
            if line:
                word = line.rstrip()
                analyses = analyzer.analyze(LexiconEntry(word))
                if not analyses:
                    continue
                total_prob = 0
                for e in analyses:
                    edge_prob = math.exp(-e.attr['cost'])
                    r_id = model.rule_set.get_id(e.rule)
                    freq_prob = norm.cdf(-e.source.logfreq, loc=means[r_id],
                                         scale=sdevs[r_id])
                    freq_logprob = norm.logcdf(-e.source.logfreq, loc=means[r_id],
                                               scale=sdevs[r_id])
#                     write_line(fp, ('---', str(e), -math.log(edge_prob),
#                                     freq_logprob))
                    total_prob += edge_prob * freq_prob
                if total_prob <= 0:
                    continue
                total_cost = -math.log(total_prob)
                if total_cost < shared.config['generate'].getfloat('max_cost'):
                    write_line(fp, (word, total_cost))
#                     print(word, total_cost)
            else:
                break

