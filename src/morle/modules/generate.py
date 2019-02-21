from morle.algorithms.analyzer import Analyzer
import morle.algorithms.fst as FST
from morle.datastruct.lexicon import Lexicon, LexiconEntry, unnormalize_word
from morle.datastruct.graph import GraphEdge
from morle.models.suite import ModelSuite
from morle.utils.files import file_exists, full_path, open_to_write
import morle.shared as shared

from collections import defaultdict
import hfst
import logging
import math
import numpy as np
from scipy.stats import norm
import subprocess
import tqdm


# TODO
# - write output to a file


def get_analyzer(filename, lexicon, model):
    if file_exists(filename):
        analyzer = Analyzer.load(filename, lexicon, model)
    else:
        analyzer = Analyzer(lexicon, model)
        analyzer.save(filename)
    return analyzer


def create_new_words_acceptor_if_not_exists(filename, analyzer, lexicon):
    if not file_exists(filename):
        new_words_acceptor = hfst.HfstTransducer(analyzer.fst)
        new_words_acceptor.convert(
            hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        new_words_acceptor.input_project()
        new_words_acceptor.minimize()
        new_words_acceptor.subtract(lexicon.to_fst())
        new_words_acceptor.minimize()
        FST.save_transducer(new_words_acceptor, filename)


def generate_words(tr_file :str, analyzer :Analyzer, model :ModelSuite,
                   freq_model :bool = True, sum_analyses :bool = True,
                   min_freq :float = 1, max_freq :float = 2):
    logging.getLogger('main').info('Precomputing the Gaussian distribution table...')
    _normcdf_cache = norm.cdf(np.array(range(-10000, 10001)) / 1000)
    max_cost = shared.config['generate'].getfloat('max_cost')

    def _normcdf(x):
        if x < -10:
            x = -10
        elif x > 10:
            x = 10
        return _normcdf_cache[int((x+10)*1000)]

    def _edge_prob_ratio(edge :GraphEdge) -> float:
        r_id = model.rule_set.get_id(edge.rule)
        prob = model.edge_model.edge_prob(edge)
        return prob / (1-prob)

    def _edge_freq_prob(edge :GraphEdge) -> float:
        r_id = model.rule_set.get_id(edge.rule)
        mean = model.edge_frequency_model.means[r_id]
        sdev = model.edge_frequency_model.sdevs[r_id]
        norm_min_freq = (log_min_freq - edge.source.logfreq - mean) / sdev
        norm_max_freq = (log_max_freq - edge.source.logfreq - mean) / sdev
        freq_prob = (_normcdf(log_max_freq) - _normcdf(log_min_freq)) / sdev
        return freq_prob
    
    logging.getLogger('main').info('Generating...')
    cmd = ['hfst-fst2strings', full_path(tr_file)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, 
                         universal_newlines=True, bufsize=1)
    log_max_freq = math.log(max_freq)
    log_min_freq = math.log(min_freq)
    while True:
        try: 
            line = p.stdout.readline().strip()
            if line:
                word = unnormalize_word(line.rstrip())
                analyses = analyzer.analyze(LexiconEntry(word), compute_cost=False)
                word_prob_ratio = 0
                for edge in analyses:
                    prob_ratio = _edge_prob_ratio(edge)
                    if freq_model:
                        prob_ratio *= _edge_freq_prob(edge)
                    if sum_analyses:
                        word_prob_ratio += prob_ratio
                    else:
                        word_prob_ratio = max(word_prob_ratio, prob_ratio)
                if word_prob_ratio > 0:
                    cost = -math.log(word_prob_ratio)
                    if cost < max_cost:
                        yield (word, cost)
            else:
                break
        except Exception as e:
            logging.getLogger('main').warning(str(e))


def run() -> None:
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    analyzer = get_analyzer('analyzer.fsm', lexicon, model)
    tr_file = 'wordgen.fst'
    create_new_words_acceptor_if_not_exists(tr_file, analyzer, lexicon)

    generator = tqdm.tqdm(generate_words(
        tr_file, analyzer, model,
        freq_model = shared.config['generate'].getboolean('freq_model'),
        sum_analyses = shared.config['generate'].getboolean('sum_analyses'),
        min_freq = shared.config['generate'].getfloat('min_freq'),
        max_freq = shared.config['generate'].getfloat('max_freq')))
    # TODO write to a file
    for word, cost in generator:
        print(word, cost, sep='\t')

