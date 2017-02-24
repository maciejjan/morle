# from algorithms.fst import binary_disjunct, delenv, delfilter,\
#                            seq_to_transducer, save_transducer
# from utils.printer import progress_printer
from utils.files import full_path, remove_file
import shared

# from collections import defaultdict
import hfst
import subprocess
# import os.path
import sys
import tqdm

def write_delenv_transducer(filename, max_affix_size, max_infix_size,\
                            max_infix_slots):

    def add_deletion_chain(outfp, state, length):
        print(state, state+1, '@_EPSILON_SYMBOL_@', '@_DELSLOT_SYMBOL_@',
              sep='\t', file=outfp)
        for i in range(1, length+1):
            print(state+i, state+i+1, '@_UNKNOWN_SYMBOL_@', 
                  '@_DELETION_SYMBOL_@', sep='\t', file=outfp)
        last_state = state + length + 1
        for i in range(length+1):
            print(state+i, last_state, '@_EPSILON_SYMBOL_@',
                  '@_EPSILON_SYMBOL_@', sep='\t', file=outfp)
        return last_state

    def add_identity_loop(outfp, state):
        print(state, state+1, '@_IDENTITY_SYMBOL_@', '@_IDENTITY_SYMBOL_@',
              sep='\t', file=outfp)
        print(state+1, state+1, '@_IDENTITY_SYMBOL_@', '@_IDENTITY_SYMBOL_@',
              sep='\t', file=outfp)
        return state+1

    with open(filename, 'w+') as outfp:
#     with open_to_write(filename) as outfp:
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
#     with open_to_write(filename) as outfp:
    with open(filename, 'w+') as outfp:
        print(0, 0, '@_DELSLOT_SYMBOL_@', '@_DELSLOT_SYMBOL_@', sep='\t',
              file=outfp)
        for i in range(length):
            print(i, i+1, '@_IDENTITY_SYMBOL_@', '@_IDENTITY_SYMBOL_@',
                  sep='\t', file=outfp)
            print(i+1, i, '@_DELETION_SYMBOL_@', '@_EPSILON_SYMBOL_@',
                  sep='\t', file=outfp)
            print(i+1, i+1, '@_DELSLOT_SYMBOL_@', '@_DELSLOT_SYMBOL_@',
                  sep='\t', file=outfp)
            print(i+1, file=outfp)
        first_negative_state = length+1
        print(0, first_negative_state, '@_DELETION_SYMBOL_@',
              '@_EPSILON_SYMBOL_@', sep='\t', file=outfp)
        print(first_negative_state, 0, '@_IDENTITY_SYMBOL_@',
              '@_IDENTITY_SYMBOL_@', sep='\t', file=outfp)
        print(first_negative_state, first_negative_state, '@_DELSLOT_SYMBOL_@', 
              '@_DELSLOT_SYMBOL_@', sep='\t', file=outfp)
        for i in range(length-1):
            print(first_negative_state+i, first_negative_state+i+1,
                  '@_DELETION_SYMBOL_@', '@_EPSILON_SYMBOL_@', sep='\t',
                  file=outfp)
            print(first_negative_state+i+1, first_negative_state+i,
                  '@_IDENTITY_SYMBOL_@', '@_IDENTITY_SYMBOL_@', sep='\t',
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
#     p.stdin.write('read att {}\n'.format(delfilter_file))
#     p.stdin.write('read att {}\n'.format(delenv_file))
#     p.stdin.write('compose\n')
#     p.stdin.write('minimize\n')
#     p.stdin.write('load stack {}\n'.format(full_path(lexicon_tr_file)))
#     p.stdin.write('compose\n')
#     p.stdin.write('minimize\n')
#     p.stdin.write('define T\n')
#     p.stdin.write('push T\n')
#     p.stdin.write('push T\n')
#     p.stdin.write('invert\n')
#     p.stdin.write('lookup-optimize\n')
#     p.stdin.write('save stack {}\n'.format(full_path('fastss.fsm')))
#     p.stdin.write('quit\n')
    p.stdin.close()
    p.wait()
    
#     remove_file(delenv_file)
#     remove_file(delfilter_file)

# def build_substring_transducer(lexicon_tr, max_word_len):
#     alphabet = lexicon_tr.get_alphabet()
#     d = delenv(alphabet,
#                shared.config['preprocess'].getint('max_affix_length'),
#                shared.config['preprocess'].getint('max_infix_length'),
#                shared.config['preprocess'].getint('max_infix_slots'))
#     d.compose(delfilter(alphabet, max_word_len))
#     d.minimize()
#     
#     result = hfst.HfstTransducer(lexicon_tr)
#     result.compose(d)
#     result.minimize()
#     return result
# 
# # TODO output file as function parameter
# # TODO use hfst-xfst and hfst-lexc
# def build_fastss_cascade(lex_tr_left, lex_tr_right=None, max_word_len=20):
#     fastss_tr_left = build_substring_transducer(lex_tr_left, max_word_len)
#     if lex_tr_right is None:
#         fastss_tr_right = hfst.HfstTransducer(fastss_tr_left)
#     else:
#         fastss_tr_right = \
#             build_substring_transducer(lex_tr_right, max_word_len)
#     fastss_tr_right.invert()
# 
#     fastss_tr_left.convert(hfst.ImplementationType.HFST_OL_TYPE)
#     fastss_tr_right.convert(hfst.ImplementationType.HFST_OL_TYPE)
# #     save_transducer(substr_tr_left,
# #                     os.path.join(shared.options['working_dir'], 'tr_l.fsm'),
# #                     type=hfst.ImplementationType.HFST_OL_TYPE)
# #     save_transducer(substr_tr_right,
# #                     os.path.join(shared.options['working_dir'], 'tr_r.fsm'),
# #                     type=hfst.ImplementationType.HFST_OL_TYPE)
#     # save cascade
#     tr_path = os.path.join(shared.options['working_dir'], 'fastss.fsm')
#     ostr = hfst.HfstOutputStream(filename=tr_path, 
#                                  type=hfst.ImplementationType.HFST_OL_TYPE)
#     try:
#         ostr.write(fastss_tr_left)
#         ostr.write(fastss_tr_right)
#         ostr.flush()
#     except Exception:
#         pass
#     finally:
#         ostr.close()

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
    istr = hfst.HfstInputStream(transducer_path)
    delenv = istr.read()
    right_tr = istr.read()
    istr.close()

    block_size = shared.config['preprocess'].getint('block_size')
    count = 0
    progressbar = None
    if shared.config['preprocess'].getint('num_processes') == 1:
        progressbar = tqdm.tqdm(len(words))
    while count < len(words):
        block = words[count:count+block_size]
#         print(block)
        tr = hfst.fst(block)
        tr.minimize()
        tr.convert(hfst.ImplementationType.SFST_TYPE)
#         print(tr.number_of_states(), tr.number_of_arcs())
        tr.compose(delenv)
        tr.minimize()
#         print(tr.number_of_states(), tr.number_of_arcs())
        tr.compose(right_tr)
        tr.minimize()
#         print(tr.number_of_states(), tr.number_of_arcs())
        similar_words = { word: set() for word in block }

        tr_b = hfst.HfstBasicTransducer(tr)
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
                        elif sym_in == hfst.IDENTITY or sym_in == hfst.UNKNOWN:
                            raise RuntimeError('Illegal symbol!')
                        sym_out = transition.get_output_symbol()
                        if sym_out == hfst.EPSILON:
                            sym_out = ''
                        elif sym_out == hfst.IDENTITY or sym_out == hfst.UNKNOWN:
                            raise RuntimeError('Illegal symbol!')
                        current_io_pairs[target_state].add(
                            (str_in+sym_in, str_out+sym_out))
            previous_io_pairs = current_io_pairs

        for word_1, word_2 in results:
            yield (word_1, word_2)
        
#         for word in block:
#             yield (word, word)

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

