from utils.files import full_path
# from utils.printer import progress_printer

import hfst
import os.path
import shared
import sys
import tqdm
import types

def seq_to_transducer(alignment, weight=0.0, type=None, alphabet=None):
    if type is None:
        type=shared.config['FST'].getint('transducer_type')
    tr = hfst.HfstBasicTransducer()
    if alphabet is None:
        alphabet = ()
    alphabet = tuple(sorted(set(alphabet) | set(sum(alignment, ()))))
    tr.add_symbols_to_alphabet(alphabet)
    last_state_id = 0
    for (x, y) in alignment:
        state_id = tr.add_state()
#        tr.add_transition(state_id, hfst.HfstBasicTransition(state_id, hfst.EPSILON, hfst.EPSILON, 0.0))
        if (x, y) == (hfst.IDENTITY, hfst.IDENTITY):
            tr.add_transition(last_state_id, 
                              hfst.HfstBasicTransition(state_id,
                                                          hfst.IDENTITY,
                                                          hfst.IDENTITY,
                                                          0.0))
            tr.add_transition(state_id, 
                              hfst.HfstBasicTransition(state_id,
                                                          hfst.IDENTITY,
                                                          hfst.IDENTITY,
                                                          0.0))
            for a in tr.get_alphabet():
                if not a.startswith('@_'):
                    tr.add_transition(last_state_id, hfst.HfstBasicTransition(state_id, a, a, 0.0))
                    tr.add_transition(state_id, hfst.HfstBasicTransition(state_id, a, a, 0.0))
        else:
            tr.add_transition(last_state_id, 
                              hfst.HfstBasicTransition(state_id, x, y, 0.0))
        last_state_id = state_id
    tr.set_final_weight(last_state_id, weight)
    return hfst.HfstTransducer(tr, type)

def binary_disjunct(transducers, print_progress=False):
    iterator, progressbar = None, None
    if isinstance(transducers, list):
        iterator = iter(transducers)
        if print_progress:
            progressbar = tqdm.tqdm(total=len(transducers))
    elif isinstance(transducers, types.GeneratorType):
        iterator = transducers
    else:
        raise TypeError('\'transducers\' must be a list or a generator!')


    stack, sizes = [], []
    count = 0
    while True:
        if len(sizes) >= 2 and sizes[-1] == sizes[-2]:
            # disjunct the two top transducers from the stack
            first, first_size = stack.pop(), sizes.pop()
            second, second_size = stack.pop(), sizes.pop()
            first.disjunct(second)
            stack.append(first)
            sizes.append(first_size + second_size)
            stack[-1].minimize()
        else:
            # push a new transducer to the stack
            try:
                stack.append(next(iterator))
                sizes.append(1)
                count += 1
                if print_progress and progressbar is not None:
                    progressbar.update()
            except StopIteration:
                break
    # disjunct the remaining transducers and minimize the result
    t = stack.pop()
    while stack:
        t.disjunct(stack.pop())
    t.determinize()
    t.minimize()
#    t.push_weights(hfst.TO_INITIAL_STATE)
    t.push_weights_to_end()
    if print_progress and progressbar is not None:
        progressbar.close()
    return t

A_TO_Z = tuple('abcdefghijklmnoprstuvwxyz')

def generate_id(id_num):
    result = A_TO_Z[id_num % len(A_TO_Z)]
    while id_num > len(A_TO_Z):
        id_num //= len(A_TO_Z)
        result = A_TO_Z[id_num % len(A_TO_Z)-1] + result
    return result

#def id_absorber(id_num):
#    seq = ('$',) + tuple(generate_id(id_num))
#    return seq_to_transducer(\
#        zip(seq, (hfst.EPSILON,)*len(seq)),\
#        alphabet=A_TO_Z + ('$',))

def id_generator():
    tr = hfst.HfstBasicTransducer()
    tr.add_symbols_to_alphabet(A_TO_Z + ('$',))
    tr.add_transition(0, 
                      hfst.HfstBasicTransition(1, '$', '$', 0.0))
    for c in A_TO_Z:
        tr.add_transition(1, 
                          hfst.HfstBasicTransition(1, c, c, 0.0))
    tr.set_final_weight(1, 0.0)
    return hfst.HfstTransducer(tr, settings.TRANSDUCER_TYPE)

def number_of_paths(transducer):
    # in n-th iteration paths_for_state[s] contains the number of paths
    # of length n terminating in state s
    # terminates if maximum n is reached, i.e. paths_for_state > 0
    # only for states without outgoing transitions
    t = hfst.HfstBasicTransducer(transducer)
    paths_for_state = [1] + [0] * (len(t.states())-1)
    result = 0
    changed = True
    while changed:
        changed = False
        new_paths_for_state = [0] * len(t.states())
        for state in t.states():
            if paths_for_state[state] > 0:
                for tr in t.transitions(state):
                    new_paths_for_state[tr.get_target_state()] +=\
                        paths_for_state[state]
                    changed = True
        for state in t.states():
            if t.is_final_state(state):
                result += new_paths_for_state[state]
        paths_for_state = new_paths_for_state
    return result

def delenv(alphabet, max_affix_size, max_infix_size, max_infix_slots,
           deletion_symbol='@_DEL_@', deletion_slot_symbol='@_DELSLOT_@'):
    
    def add_deletion_chain(tr, alphabet, state, length):
        tr.add_transition(state,
                          hfst.HfstBasicTransition(
                              state+1, hfst.EPSILON, deletion_slot_symbol, 0.0))
        for i in range(1, length+1):
            for c in alphabet:
                if c not in (hfst.EPSILON, hfst.IDENTITY, hfst.UNKNOWN):
                    tr.add_transition(state+i,
                                      hfst.HfstBasicTransition(
                                          state+i+1, 
                                          c, deletion_symbol, 0.0))
        last_state = state + length + 1
        for i in range(length+1):
            tr.add_transition(state+i,
                              hfst.HfstBasicTransition(
                                  last_state,
                                  hfst.EPSILON, hfst.EPSILON, 0.0))
        return last_state

    def add_identity_loop(tr, alphabet, state):
        for c in alphabet:
            if c not in (hfst.EPSILON, hfst.IDENTITY, hfst.UNKNOWN):
                tr.add_transition(state,
                                  hfst.HfstBasicTransition(state+1, c, c, 0.0))
                tr.add_transition(state+1,
                                  hfst.HfstBasicTransition(state+1, c, c, 0.0))
        return state+1

    tr = hfst.HfstBasicTransducer()
    # prefix
    state = add_deletion_chain(tr, alphabet, 0, max_affix_size)
    state = add_identity_loop(tr, alphabet, state)
    # infixes
    for i in range(max_infix_slots):
        state = add_deletion_chain(tr, alphabet, state, max_infix_size)
        state = add_identity_loop(tr, alphabet, state)
    # suffix
    state = add_deletion_chain(tr, alphabet, state, max_affix_size)
    tr.set_final_weight(state, 0.0)
    tr_c = hfst.HfstTransducer(tr)
    tr_c.remove_epsilons()
    tr_c.minimize()
    return tr_c
    
# TODO similar_words():
#      lookup word to find out substrings,
#      lookup each substring, sum and remove duplicates
def delfilter(alphabet, length, deletion_symbol='@_DEL_@',
              deletion_slot_symbol='@_DELSLOT_@'):
    tr = hfst.HfstBasicTransducer()
    tr.set_final_weight(0, 0.0)
    tr.add_transition(0,
                      hfst.HfstBasicTransition(
                          0, deletion_slot_symbol, deletion_slot_symbol, 0.0))
    printable_chars = set(alphabet) -\
                      { hfst.EPSILON, hfst.IDENTITY, hfst.UNKNOWN,
                        deletion_symbol }
    for i in range(length):
        for c in printable_chars:
            tr.add_transition(i,
                              hfst.HfstBasicTransition(i+1, c, c, 0.0))
        tr.add_transition(i+1,
                          hfst.HfstBasicTransition(
                              i, deletion_symbol, hfst.EPSILON, 0.0))
        tr.add_transition(i+1,
                          hfst.HfstBasicTransition(
                              i+1, deletion_slot_symbol, deletion_slot_symbol, 0.0))
        tr.set_final_weight(i+1, 0.0)
    first_negative_state = length+1
    tr.add_transition(0, hfst.HfstBasicTransition(
                             first_negative_state, deletion_symbol,
                             hfst.EPSILON, 0.0))
    for c in printable_chars:
        tr.add_transition(first_negative_state, 
                          hfst.HfstBasicTransition(0, c, c, 0.0))
    for i in range(length-1):
        tr.add_transition(first_negative_state+i,
                          hfst.HfstBasicTransition(
                              first_negative_state+i+1, 
                              deletion_symbol, hfst.EPSILON, 0.0))
        tr.add_transition(first_negative_state+i+1,
                          hfst.HfstBasicTransition(
                              first_negative_state+i+1, deletion_slot_symbol, deletion_slot_symbol, 0.0))
        for c in printable_chars:
            tr.add_transition(first_negative_state+i+1,
                              hfst.HfstBasicTransition(
                                  first_negative_state+i, c, c, 0.0))
    tr_c = hfst.HfstTransducer(tr)
    return tr_c
                                                   

def rootgen_transducer(rootdist):
    # create an automaton for word generation
    if shared.config['Features'].getint('rootdist_n') != 1:
        raise NotImplementedError('Not implemented for rootdist_n != 1')
    weights = rootdist.features[0].log_probs
    tr = hfst.HfstBasicTransducer()
    tr.set_final_weight(0, weights[('#',)])
    for char, weight in weights.items():
        if char != ('#',):
            tr.add_transition(0, 
                hfst.HfstBasicTransition(0, char[0], char[0], weight))
    return hfst.HfstTransducer(tr)

def tag_absorber(alphabet):
    tr = hfst.HfstBasicTransducer()
    for c in alphabet:
        if shared.compiled_patterns['symbol'].match(c):
            tr.add_transition(0,
                hfst.HfstBasicTransition(0, c, c, 0.0))
        elif shared.compiled_patterns['tag'].match(c):
            tr.add_transition(0,
                hfst.HfstBasicTransition(1, c, hfst.EPSILON, 0.0))
            tr.add_transition(1,
                hfst.HfstBasicTransition(1, c, hfst.EPSILON, 0.0))
    tr.set_final_weight(0, 0.0)
    tr.set_final_weight(1, 0.0)
    return hfst.HfstTransducer(tr)

def tag_acceptor(tag, alphabet):
    tr = hfst.HfstBasicTransducer()
    for c in alphabet:
        if shared.compiled_patterns['symbol'].match(c):
            tr.add_transition(0,
                hfst.HfstBasicTransition(0, c, c, 0.0))
    tr.set_final_weight(0, 0.0)
    tr_c = hfst.HfstTransducer(tr)
    tr_c.concatenate(seq_to_transducer(tuple(zip(tag, tag))))
    return tr_c

def load_transducer(filename):
    path = os.path.join(shared.options['working_dir'], filename)
    istr = hfst.HfstInputStream(path)
    transducer = istr.read()
    istr.close()
    return transducer

def load_cascade(filename):
    transducers = []
    istr = hfst.HfstInputStream(full_path(filename))
    while not istr.is_eof():
        transducers.append(istr.read())
    istr.close()
    return tuple(transducers)

def save_transducer(transducer, filename):
    path = os.path.join(shared.options['working_dir'], filename)
    ostr = hfst.HfstOutputStream(filename=path, type=transducer.get_type())
    ostr.write(transducer)
    ostr.flush()
    ostr.close()

def save_cascade(transducers, filename, type=None):
    raise NotImplementedError()
#     istr = hfst.HfstInputStream(filename)
#     delenv = istr.read()
#     right_tr = istr.read()
#     istr.close()
#     return delenv, right_tr
