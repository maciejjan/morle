from algorithms.fst import binary_disjunct, delenv, delfilter, seq_to_transducer, save_transducer
from utils.printer import progress_printer
import shared

from collections import defaultdict
import hfst
import subprocess
import os.path
import sys

def build_substring_transducer(lexicon_tr, max_word_len):
    alphabet = lexicon_tr.get_alphabet()
    d = delenv(alphabet,
               shared.config['preprocess'].getint('max_affix_length'),
               shared.config['preprocess'].getint('max_infix_length'),
               shared.config['preprocess'].getint('max_infix_slots'))
    d.compose(delfilter(alphabet, max_word_len))
    d.minimize()
    
    result = hfst.HfstTransducer(lexicon_tr)
    result.compose(d)
    result.minimize()
    return result

# TODO output file as function parameter
def build_fastss_cascade(lex_tr_left, lex_tr_right=None, max_word_len):
    fastss_tr_left = build_substring_transducer(lex_tr_left, max_word_len)
    if lex_tr_right is None:
        fastss_tr_right = hfst.HfstTransducer(fastss_tr_left)
    else:
        fastss_tr_right = build_substring_transducer(lex_tr_right, max_word_len)
    fastss_tr_right.invert()

    fastss_tr_left.convert(hfst.ImplementationType.HFST_OL_TYPE)
    fastss_tr_right.convert(hfst.ImplementationType.HFST_OL_TYPE)
#     save_transducer(substr_tr_left,
#                     os.path.join(shared.options['working_dir'], 'tr_l.fsm'),
#                     type=hfst.ImplementationType.HFST_OL_TYPE)
#     save_transducer(substr_tr_right,
#                     os.path.join(shared.options['working_dir'], 'tr_r.fsm'),
#                     type=hfst.ImplementationType.HFST_OL_TYPE)
    # save cascade
    tr_path = os.path.join(shared.options['working_dir'], 'fastss.fsm')
    ostr = hfst.HfstOutputStream(filename=tr_path, 
                                 type=hfst.ImplementationType.HFST_OL_TYPE)
    try:
        ostr.write(fastss_tr_left)
        ostr.write(fastss_tr_right)
        ostr.flush()
    except Exception:
        pass
    finally:
        ostr.close()

# def lookup_subprocess(words, transducer_path):
#     cmd = ['hfst-lookup', '-i', transducer_path, '-C', 'composition']
#     p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                          stderr=subprocess.DEVNULL, universal_newlines=True,
#                          bufsize=1)
#     pout, perr = p.communicate('\n'.join(words)+'\n')
#     return pout

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
    p.stdout.close()
    p.wait()

# TODO:
# - start multiple subprocesses in parallel and return their outputs

