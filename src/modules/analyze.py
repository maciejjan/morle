from algorithms.analyzer import Analyzer
from datastruct.lexicon import Lexicon
from models.suite import ModelSuite
import shared


def run():
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    kwargs = {}
    kwargs['predict_vec'] = shared.config['analyze'].getboolean('predict_vec')
    analyzer = Analyzer(lexicon, model, **kwargs)
    lexicon_to_analyze = Lexicon.load(shared.filenames['analyze.wordlist'])
    for lexitem in lexicon_to_analyze:
        analyses = analyzer.analyze(lexitem)
        for a in analyses:
            print(a.source, a.target, a.rule, a.attr['cost'],
                  a.target.vec if kwargs['predict_vec'] else '')

