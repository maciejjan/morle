# from algorithms.fst import binary_disjunct, delenv, delfilter,\
#                            seq_to_transducer, save_transducer
# from utils.printer import progress_printer
import shared

# from collections import defaultdict
# import hfst
import subprocess
# import os.path
import sys

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

def build_fastss_cascade(tr_file, max_word_len=20):
    # TODO build delfilter and delenv
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=None, universal_newlines=True)
    p.stdin.write('read att delfilter.att\n')
    p.stdin.write('read att delenv.att\n')
    p.stdin.write('compose\n')
    p.stdin.write('minimize\n')
    p.stdin.write('read lexc {}\n'.format(filename + '.lex'))
    p.stdin.write('compose\n')
    p.stdin.write('minimize\n')
    p.stdin.write('define T\n')
    p.stdin.write('push T\n')
    p.stdin.write('push T\n')
    p.stdin.write('invert\n')
    p.stdin.write('lookup-optimize\n')
    p.stdin.write('save stack {}\n'.format(filename + '.fastss.fsm'))
    p.stdin.write('quit\n')
    p.stdin.close()
    p.wait()

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

# TODO parameter `restart_interval` -- restart the subprocess every N words
#      to counter the memory leak
def similar_words(words, transducer_path):
    cmd = ['hfst-lookup', '-i', transducer_path, '-C', 'composition']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, universal_newlines=True,
                         bufsize=1)
    count = 0
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
    p.stdin.close()
    p.wait()

