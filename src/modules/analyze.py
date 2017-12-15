from algorithms.analyzer import Analyzer
from datastruct.lexicon import Lexicon
from models.suite import ModelSuite
import shared


def run():
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    kwargs = {}
    kwargs['predict_vec'] = shared.config['analyze'].getboolean('predict_vec')
    kwargs['max_results'] = shared.config['analyze'].getint('max_results')
    analyzer = Analyzer(lexicon, model, **kwargs)
    lexicon_to_analyze = Lexicon.load(shared.filenames['analyze.wordlist'])
    for lexitem in lexicon_to_analyze:
        analyses = analyzer.analyze(lexitem)
        for a in analyses:
            vec_str = ' '.join(list(map(str, a.target.vec))) \
                      if kwargs['predict_vec'] \
                      else ''
            print(a.source, a.target, a.rule, a.attr['cost'], vec_str,
                  sep='\t')
        # TODO
        # - max_results parameter
        # - including the analysis of a word as a root

