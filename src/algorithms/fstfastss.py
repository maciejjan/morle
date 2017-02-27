from algorithms.fst import load_cascade
from utils.files import full_path, remove_file
import shared

import hfst
import subprocess
import sys
import tqdm

def write_delenv_transducer(filename, max_affix_size, max_infix_size,\
                            max_infix_slots):

    def add_deletion_chain(outfp, state, length):
        print(state, state+1, hfst.EPSILON, '@_DELSLOT_SYMBOL_@',
              sep='\t', file=outfp)
        for i in range(1, length+1):
            print(state+i, state+i+1, hfst.UNKNOWN, 
                  '@_DELETION_SYMBOL_@', sep='\t', file=outfp)
        last_state = state + length + 1
        for i in range(length+1):
            print(state+i, last_state, hfst.EPSILON,
                  hfst.EPSILON, sep='\t', file=outfp)
        return last_state

    def add_identity_loop(outfp, state):
        print(state, state+1, hfst.IDENTITY, hfst.IDENTITY,
              sep='\t', file=outfp)
        print(state+1, state+1, hfst.IDENTITY, hfst.IDENTITY,
              sep='\t', file=outfp)
        return state+1

    with open(filename, 'w+') as outfp:
        # prefix
        state = add_deletion_chain(outfp, 0, max_affix_size)
        state = add_identity_loop(outfp, state)
        # infixes
        for i in range(max_infix_slots):
            state = add_deletion_chain(outfp, state, max_infix_size)
            state = add_identity_loop(outfp, state)
        # suffix
        state = add_deletion_chain(outfp, state, max_affix_size)
        # set final state
        print(state, file=outfp)

def write_delfilter_transducer(filename, length):
    with open(filename, 'w+') as outfp:
        print(0, 0, '@_DELSLOT_SYMBOL_@', '@_DELSLOT_SYMBOL_@', sep='\t',
              file=outfp)
        for i in range(length):
            print(i, i+1, hfst.IDENTITY, hfst.IDENTITY,
                  sep='\t', file=outfp)
            print(i+1, i, '@_DELETION_SYMBOL_@', hfst.EPSILON,
                  sep='\t', file=outfp)
            print(i+1, i+1, '@_DELSLOT_SYMBOL_@', '@_DELSLOT_SYMBOL_@',
                  sep='\t', file=outfp)
            print(i+1, file=outfp)
        first_negative_state = length+1
        print(0, first_negative_state, '@_DELETION_SYMBOL_@',
              hfst.EPSILON, sep='\t', file=outfp)
        print(first_negative_state, 0, hfst.IDENTITY,
              hfst.IDENTITY, sep='\t', file=outfp)
        print(first_negative_state, first_negative_state, '@_DELSLOT_SYMBOL_@', 
              '@_DELSLOT_SYMBOL_@', sep='\t', file=outfp)
        for i in range(length-1):
            print(first_negative_state+i, first_negative_state+i+1,
                  '@_DELETION_SYMBOL_@', hfst.EPSILON, sep='\t',
                  file=outfp)
            print(first_negative_state+i+1, first_negative_state+i,
                  hfst.IDENTITY, hfst.IDENTITY, sep='\t',
                  file=outfp)
            print(first_negative_state+i+1, first_negative_state+i+1,
                  '@_DELSLOT_SYMBOL_@', '@_DELSLOT_SYMBOL_@', sep='\t',
                  file=outfp)

def build_fastss_cascade(lexicon_tr_file, max_word_len=20):
    delenv_file = full_path('delenv.att')
    write_delenv_transducer(
        delenv_file,
        shared.config['preprocess'].getint('max_affix_length'),
        shared.config['preprocess'].getint('max_infix_length'),
        shared.config['preprocess'].getint('max_infix_slots'))
    delfilter_file = full_path('delfilter.att')
    write_delfilter_transducer(delfilter_file, max_word_len)

    cmd = ['hfst-xfst', '-f', 'sfst']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=None, universal_newlines=True)
    p.stdin.write('read att {}\n'.format(delfilter_file))
    p.stdin.write('read att {}\n'.format(delenv_file))
    p.stdin.write('compose\n')
    p.stdin.write('minimize\n')
    p.stdin.write('define T\n')
    p.stdin.write('push T\n')
    p.stdin.write('load stack {}\n'.format(full_path(lexicon_tr_file)))
    p.stdin.write('compose\n')
    p.stdin.write('minimize\n')
    p.stdin.write('invert\n')
    p.stdin.write('push T\n')
    p.stdin.write('rotate stack\n')
#     p.stdin.write('lookup-optimize\n')
    p.stdin.write('save stack {}\n'.format(full_path('fastss.fsm')))
    p.stdin.write('quit\n')
    p.stdin.close()
    p.wait()
    
#     remove_file(delenv_file)
#     remove_file(delfilter_file)

def similar_words_with_lookup(words, transducer_path):
    cmd = ['hfst-lookup', '-i', transducer_path, '-C', 'composition']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, universal_newlines=True,
                         bufsize=1)
    count = 0
    restart_interval = \
        shared.config['preprocess'].getint('hfst_restart_interval')
    for word in words:
        p.stdin.write(word+'\n')
        p.stdin.flush()
        similar_words = set()
        while True:
            line = p.stdout.readline().strip()
            if line:
                cols = line.split('\t')
                if len(cols) == 3 and cols[2].startswith('0'):
                    similar_words.add(cols[1])
            else:
                break
        for sim_word in similar_words:
            yield (word, sim_word)
        count += 1
        if restart_interval > 0 and count % restart_interval == 0:
            # restart the HFST subprocess to counter the memory leak
            p.stdin.close()
            p.wait()
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, 
                                 universal_newlines=True, bufsize=1)
    p.stdin.close()
    p.wait()

def similar_words_with_block_composition(words, transducer_path):
    def _compose_block(block, delenv, right_tr, tokenizer):
#         tr = hfst.fst(block)    # TODO tokenize?
        tr = hfst.empty_fst()
        for word in block:
            tr.disjunct(hfst.tokenized_fst(tokenizer.tokenize(word)))
        tr.minimize()
        tr.convert(hfst.ImplementationType.SFST_TYPE)
#         print(tr.number_of_states(), tr.number_of_arcs())
        tr.compose(delenv)
        tr.minimize()
#         print(tr.number_of_states(), tr.number_of_arcs())
        tr.compose(right_tr)
        tr.minimize()
        return tr

    def _extract_unique_io_pairs(transducer):
        tr_b = hfst.HfstBasicTransducer(transducer)
        previous_io_pairs = []
        for s in tr_b.states():
            previous_io_pairs.append(set())
        previous_io_pairs[0].add(('', ''))
        results = set()
        empty = False
        while not empty:
            empty = True
            current_io_pairs = []
            for s in tr_b.states():
                current_io_pairs.append(set())
            for state, state_io_pairs in enumerate(previous_io_pairs):
                if state_io_pairs:
                    empty = False
                if tr_b.is_final_state(state):
                    results |= state_io_pairs
                for str_in, str_out in state_io_pairs:
                    for transition in tr_b.transitions(state):
                        target_state = transition.get_target_state()
                        sym_in = transition.get_input_symbol()
                        if sym_in == hfst.EPSILON:
                            sym_in = ''
                        elif sym_in in (hfst.IDENTITY, hfst.UNKNOWN):
                            raise RuntimeError('Illegal symbol!')
                        sym_out = transition.get_output_symbol()
                        if sym_out == hfst.EPSILON:
                            sym_out = ''
                        elif sym_out in (hfst.IDENTITY, hfst.UNKNOWN):
                            raise RuntimeError('Illegal symbol!')
                        current_io_pairs[target_state].add(
                            (str_in+sym_in, str_out+sym_out))
            previous_io_pairs = current_io_pairs
        return results

    delenv, right_tr = load_cascade(transducer_path)
    tok = hfst.HfstTokenizer()
    for sym in shared.multichar_symbols:
        tok.add_multichar_symbol(sym)
    block_size = shared.config['preprocess'].getint('block_size')
    count = 0
    progressbar = None
    if shared.config['preprocess'].getint('num_processes') == 1:
        progressbar = tqdm.tqdm(total=len(words))
    while count < len(words):
        block = words[count:count+block_size]
        tr = _compose_block(block, delenv, right_tr, tok)
        for word_1, word_2 in _extract_unique_io_pairs(tr):
            yield (word_1, word_2)
        count += block_size
        if progressbar is not None:
            progressbar.update(len(block))

def similar_words(words, transducer_path):
    method = shared.config['preprocess'].get('method')
    if method == 'lookup':
        return similar_words_with_lookup(words, transducer_path)
    elif method == 'block_composition':
        return similar_words_with_block_composition(words, transducer_path)
    else:
        raise RuntimeError('Unknown preprocessing method: {}'.format(method))

