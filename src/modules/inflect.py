import shared

import tqdm


def run():
    rules_tr = algorithms.fst.load_transducer(shared.filenames['rules-tr'])
    for lemma, tag in read_tsv_file(shared.filenames['analyze.wordlist']):
        print(lemma, tag)
        pass
    # TODO input format:
    # base<tag> tab <tag>
    # output format:
    # base<tag> tab word<tag>
    # TODO flow:
    # - create a transducer: base .o. rules .o. (universal_letter_acceptor . tag_acceptor(tag))
    # - lookup the best path in the transducer
    pass

