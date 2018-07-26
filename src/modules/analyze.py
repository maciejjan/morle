from algorithms.analyzer import Analyzer
from datastruct.lexicon import Lexicon, load_raw_vocabulary
from models.suite import ModelSuite
import shared

import tqdm


def run():
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    kwargs = {}
    kwargs['predict_vec'] = shared.config['analyze'].getboolean('predict_vec')
    kwargs['max_results'] = shared.config['analyze'].getint('max_results')
    kwargs['enable_back_formation'] = shared.config['analyze'].getboolean('enable_back_formation')
    analyzer = Analyzer(lexicon, model, **kwargs)
    lexicon_to_analyze = \
        load_raw_vocabulary(shared.filenames['analyze.wordlist'])
    for lexitem in tqdm.tqdm(lexicon_to_analyze):
        analyses = analyzer.analyze(lexitem)
        for a in analyses:
            # TODO!!! vector transposition elsewhere
            vec_str = ' '.join(map(str, map(float, list(a.target.vec.T)))) \
                      if kwargs['predict_vec'] \
                      else ''
            print(a.source, a.target, a.rule, a.attr['cost'], vec_str,
                  sep='\t')
        # TODO
        # - including the analysis of a word as a root

