from algorithms.analyzer import Analyzer
from datastruct.lexicon import Lexicon
import shared


def run():
    lexicon_to_analyze = Lexicon.load(shared.filenames['analyze.wordlist'])
    analyzer = Analyzer()
    for lexitem in lexicon_to_analyze:
        analyses = analyzer.analyze(lexitem)
        for a in analyses:
            print(a.source, a.target, a.rule, a.attr['cost'])

