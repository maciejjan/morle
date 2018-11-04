import algorithms.fst
from datastruct.lexicon import tokenize_word, normalize_seq, normalize_word, \
                               unnormalize_word
from utils.files import read_tsv_file
import shared

import hfst
from operator import itemgetter
import sys
import tqdm


# TODO only works with SimpleEdgeModel! (no rescoring)
# -> rescoring to make it work with other edge models


def run():
    lexicon_tr = algorithms.fst.load_transducer(shared.filenames['lexicon-tr'])
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    rules_tr.convert(hfst.ImplementationType.HFST_OLW_TYPE)
    alphabet = lexicon_tr.get_alphabet()
    pairs = [(lemma, tag) \
               for lemma, tag in \
                   read_tsv_file(shared.filenames['analyze.wordlist'])]
    for lemma, tag in tqdm.tqdm(pairs):
        try:
            lookup_results = \
                sorted(rules_tr.lookup(normalize_word(lemma)), key=itemgetter(1))
            found = False
            for word, cost in lookup_results:
                word_tag = ''.join(tokenize_word(word)[1])
                if word_tag == tag:
                    print(lemma, unnormalize_word(word), cost, sep='\t')
                    found = True
                    break
            if not found:
                print(lemma, '---'+tag, '---', sep='\t')
        except Exception as e:
            print(str(e), file=sys.stderr)
            print(lemma, '---'+tag, '---', sep='\t')

