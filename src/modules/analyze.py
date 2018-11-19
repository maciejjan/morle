from algorithms.analyzer import get_analyzer
from datastruct.lexicon import Lexicon, LexiconEntry, load_raw_vocabulary, \
                               unnormalize_word
from models.suite import ModelSuite
from utils.files import file_exists
import shared
import sys

import logging
import tqdm


def analyze_word(lexitem :LexiconEntry, analyzer, **kwargs):
    results = []
    analyses = analyzer.analyze(lexitem)
    predict_vec = 'predict_vec' in kwargs and kwargs['predict_vec']
    for a in analyses:
        # TODO!!! vector transposition elsewhere
        vec_str = ' '.join(map(str, map(float, list(a.target.vec.T)))) \
                  if predict_vec \
                  else ''
        src = unnormalize_word(a.source.literal) \
              if a.source is not None else ''
        tgt = unnormalize_word(a.target.literal)
        rule = str(a.rule) if a.rule is not None else ''
        results.append((src, tgt, rule, a.attr['cost'], vec_str))
    return results


def run():
    lexicon = Lexicon.load(shared.filenames['wordlist'])
    model = ModelSuite.load()
    analyzer = get_analyzer('analyzer.fsm', lexicon, model)
    predict_vec = shared.config['analyze'].getboolean('predict_vec')
    if shared.options['interactive']:
        for line in sys.stdin:
            try:
                lexitem = LexiconEntry(line.strip())
                for analysis in analyze_word(lexitem, analyzer, predict_vec=predict_vec):
                    print(*analysis, sep='\t')
            except Exception as e:
                logging.getLogger('main').warning(e)
    else:
        lexicon_to_analyze = \
            load_raw_vocabulary(shared.filenames['analyze.wordlist'])
        for lexitem in tqdm.tqdm(lexicon_to_analyze):
            for analysis in analyze_word(lexitem, analyzer, predict_vec=predict_vec):
                print(*analysis, sep='\t')

