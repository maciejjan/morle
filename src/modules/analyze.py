from algorithms.analyzer import Analyzer
from datastruct.lexicon import Lexicon, load_raw_vocabulary, unnormalize_word
from models.suite import ModelSuite
from utils.files import file_exists
import shared

import tqdm


def get_analyzer(filename, lexicon, model):
    kwargs = {}
    kwargs['predict_vec'] = \
        shared.config['analyze'].getboolean('predict_vec')
    kwargs['max_results'] = shared.config['analyze'].getint('max_results')
    kwargs['enable_back_formation'] = \
        shared.config['analyze'].getboolean('enable_back_formation')
    if file_exists(filename):
        analyzer = Analyzer.load(filename, lexicon, model, **kwargs)
    else:
        analyzer = Analyzer(lexicon, model, **kwargs)
        analyzer.save(filename)
    return analyzer


def run():
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    analyzer = get_analyzer('analyzer.fsm', lexicon, model)
    predict_vec = shared.config['analyze'].getboolean('predict_vec')
    lexicon_to_analyze = \
        load_raw_vocabulary(shared.filenames['analyze.wordlist'])
    for lexitem in tqdm.tqdm(lexicon_to_analyze):
        analyses = analyzer.analyze(lexitem)
        for a in analyses:
            # TODO!!! vector transposition elsewhere
            vec_str = ' '.join(map(str, map(float, list(a.target.vec.T)))) \
                      if predict_vec \
                      else ''
            src = unnormalize_word(a.source.literal) \
                  if a.source is not None else ''
            tgt = unnormalize_word(a.target.literal)
            rule = str(a.rule) if a.rule is not None else ''
            print(src, tgt, rule, a.attr['cost'], vec_str, sep='\t')
        # TODO
        # - including the analysis of a word as a root

