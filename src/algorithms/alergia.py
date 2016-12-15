'''ALERGIA algorithm for learning deterministic automata.'''

from collections import defaultdict
import hfst
import math
from operator import itemgetter
import re
import sys


def prefix_tree_acceptor(seqs):
    '''Returns: HfstBasicTransducer accepting all seqs.'''

    def pta_insert(automaton, seq, freq=1, state=0):
        if not seq:
            weight = 0.0 
            try:
                weight = automaton.get_final_weight(state)
            except hfst.exceptions.StateIsNotFinalException:
                pass
            automaton.set_final_weight(state, weight + freq)
            return
        target_state, weight, found = None, 0, False
        for tr in automaton.transitions(state):
            if tr.get_input_symbol() == seq[0]:
                target_state = tr.get_target_state()
                weight = tr.get_weight()
                automaton.remove_transition(state, tr)
                break
        if target_state is None:
            target_state = automaton.add_state()
        automaton.add_transition(
          state,
          hfst.HfstBasicTransition(target_state, seq[0], 
                                      seq[0], weight+freq))
        pta_insert(automaton, seq[1:], freq=freq, state=target_state)

    automaton = hfst.HfstBasicTransducer()
    for i, (seq, freq) in enumerate(seqs):
#        pta_insert(automaton, seq, freq)
        pta_insert(automaton, seq, 1)
#         sys.stdout.write('\r' + str(i))
#     print()
    return automaton

def normalize_weights(automaton):
    '''Convert frequency weights to log-probabilities.'''
    new_automaton = hfst.HfstBasicTransducer()
    queue = [0]
    processed = set()
    while queue:
        state = queue.pop()
        processed.add(state)
        transitions = list(automaton.transitions(state))
        sum_weights = sum(tr.get_weight() for tr in transitions) +\
            (automaton.get_final_weight(state)\
               if automaton.is_final_state(state) else 0)
        if automaton.is_final_state(state):
            new_final_weight = -math.log(
              automaton.get_final_weight(state) / sum_weights)
            new_automaton.set_final_weight(state, new_final_weight)
        for tr in transitions:
            new_weight = -math.log(tr.get_weight() / sum_weights)
            new_automaton.add_transition(
              state,
              hfst.HfstBasicTransition(
                tr.get_target_state(),
                tr.get_input_symbol(),
                tr.get_output_symbol(),
                new_weight))
            if tr.get_target_state() not in processed and\
                    tr.get_target_state() not in queue:
                queue.append(tr.get_target_state())
    return new_automaton

def alergia(automaton, alpha=0.05, freq_threshold=1):
    
    def _test(f1, n1, f2, n2):
        diff = abs(f1/n1 - f2/n2)
        threshold = (1/math.sqrt(n1) + 1/math.sqrt(n2)) * math.sqrt(0.5*math.log(2/alpha))
#        print('%d/%d : %d/%d ->' % (f1, n1, f2, n2), diff, ':', threshold)
        return diff < threshold

    def _get_state_weights(state):
        weights = defaultdict(lambda : 0)
        if automaton.is_final_state(state):
            weights[hfst.EPSILON] = automaton.get_final_weight(state)
        for tr in automaton.transitions(state):
            weights[tr.get_input_symbol()] = tr.get_weight()
        return weights

    def _state_sum_freq(state):
        result = 0.0
        if automaton.is_final_state(state):
            result += automaton.get_final_weight(state)
        for tr in automaton.transitions(state):  
            result += tr.get_weight()
        return result

    def _transitions_dict(state):
        return { tr.get_input_symbol() : tr
                 for tr in automaton.transitions(state) }

    def _init_parents():
        parent = [(None, None)] * len(automaton.states())
        for s in automaton.states():
            for tr in automaton.transitions(s):
                parent[tr.get_target_state()] = (s, tr)
        return parent

    def _states_topological_ordering():
        ordering = [-1] * len(automaton.states())
        queue = [0]
        i = 0
        while queue:
            state = queue.pop(0)
            queue.extend([tr.get_target_state() 
                          for tr in automaton.transitions(state)])
            ordering[state] = i
            i += 1
        return ordering

    def _compatible(state_1, state_2):
        n_1 = sum(tr.get_weight() for tr in automaton.transitions(state_1)) +\
                (0 if not automaton.is_final_state(state_1)\
                      else automaton.get_final_weight(state_1))
        n_2 = sum(tr.get_weight() for tr in automaton.transitions(state_2)) +\
                (0 if not automaton.is_final_state(state_2)\
                      else automaton.get_final_weight(state_2))
        weights_1 = _get_state_weights(state_1)
        weights_2 = _get_state_weights(state_2)
        for key in set(weights_1.keys()) | set(weights_2.keys()):
            if not _test(weights_1[key], n_1, weights_2[key], n_2):
                return False
        return True

    def _merge(q_r, q_b):
#        print('merge(%d, %d)' % (q_r, q_b))
        q_f, tr = parent[q_b]
#        print(q_f)
        symbol = tr.get_input_symbol()
        weight = tr.get_weight()
        automaton.remove_transition(q_f, tr)
        parent[q_b] = (None, None)
        automaton.add_transition(
          q_f,
          hfst.HfstBasicTransition(q_r, symbol, symbol, weight))
        _fold(q_r, q_b)

    def _fold(q_r, q_b):
        if automaton.is_final_state(q_b):
            weight = automaton.get_final_weight(q_b)
            if automaton.is_final_state(q_r):
                weight += automaton.get_final_weight(q_r)
            automaton.set_final_weight(q_r, weight)
        trs_r = _transitions_dict(q_r)
        trs_b = _transitions_dict(q_b)
        for key, tr_b in trs_b.items():
            if key in trs_r:
                automaton.remove_transition(q_r, trs_r[key])
                automaton.add_transition(
                  q_r,
                  hfst.HfstBasicTransition(
                    trs_r[key].get_target_state(), key, key,
                    trs_r[key].get_weight() + tr_b.get_weight()))
                _fold(trs_r[key].get_target_state(), tr_b.get_target_state())
            else:
                automaton.remove_transition(q_b, tr_b)
                tr_r = hfst.HfstBasicTransition(
                         tr_b.get_target_state(), key, key,
                         tr_b.get_weight())
                automaton.add_transition(q_r, tr_r)
                parent[tr_b.get_target_state()] = (q_r, tr_r)

    def _remove_unreachable_states(automaton):
#        global automaton
        new_automaton = hfst.HfstBasicTransducer()
        state_map = {0: 0}
        queue = [0]
        processed = set()
        while queue:
            state = queue.pop()
            processed.add(state)
            if automaton.is_final_state(state):
                new_automaton.set_final_weight(
                  state_map[state], automaton.get_final_weight(state))
            for tr in automaton.transitions(state):
                if tr.get_target_state() not in state_map:
                    state_map[tr.get_target_state()] = new_automaton.add_state()
                new_automaton.add_transition(
                  state_map[state],
                  hfst.HfstBasicTransition(
                    state_map[tr.get_target_state()],
                    tr.get_input_symbol(),
                    tr.get_output_symbol(),
                    tr.get_weight()))
                if tr.get_target_state() not in processed and\
                        tr.get_target_state() not in queue:
                    queue.append(tr.get_target_state())
        return new_automaton


    def _choose(states):
        states_fil = [q for q in states\
                             if _state_sum_freq(q) > freq_threshold]
        if not states_fil:
            return None
        return min(states_fil, key=lambda q: states_ord[q])

    red_states = {0}
    blue_states = { tr.get_target_state() 
                    for tr in automaton.transitions(0) }
    states_ord = _states_topological_ordering()
    parent = _init_parents()

    while blue_states:
        q_b = _choose(blue_states)
        if q_b is None:
            break
        blue_states.remove(q_b)
#        print(q_b, q_b in blue_states)
        merged = False
        for q_r in red_states:
            if _compatible(q_r, q_b):
                _merge(q_r, q_b)
#                print(q_b, q_b in blue_states)
                merged = True
                break
        if not merged:
            red_states.add(q_b)
        blue_states = { tr.get_target_state() 
                          for q in red_states
                          for tr in automaton.transitions(q) } -\
                      red_states
#        print(q_b, q_b in blue_states)
#        print()
    automaton = normalize_weights(_remove_unreachable_states(automaton))
    return automaton

def validate(automaton, seqs):
    '''Test for an independent set of words whether all probabilities > 0.'''
    passed, total = 0, 0
    with open('alergia-validation.txt', 'w+') as fp:
        for seq, freq in seqs:
#        print(''.join(seq))
#        if not automaton.lookup(''.join(seq)):
            total += 1
            fp.write(''.join(seq) + '\t' + str(automaton.lookup(seq)) + '\n')
            if automaton.lookup(seq):
                passed += 1
            else:
#            raise Exception('Validation failed on %s' % ''.join(seq))
                print('Validation failed on %s' % ''.join(seq))
    print('Validation passed for %d/%d words.' % (passed, total))

def print_transitions(automaton, state):
    transitions = [(tr.get_target_state(), 
                    tr.get_input_symbol(), 
                    tr.get_output_symbol(), 
                    tr.get_weight()) for tr in automaton.transitions(state)]
    for tr in sorted(transitions, key=itemgetter(1)):
        print(*tr)
    
# print('Loading training data...')
# seqs = load_seqs_from_file(TRAINING_FILE)
# 
# print('Building tree acceptor...')
# automaton = prefix_tree_acceptor(seqs)
# 
# print('Size before learning:', len(automaton.states())) 
# 
# print('Running ALERGIA...')
# automaton = alergia(automaton, alpha=ALPHA)
# 
# print('Size after learning: ', len(automaton.states()))
# #TODO save the resulting automaton

#print('Converting...')
#tr = hfst.HfstTransducer(automaton)
#tr.convert(hfst.HFST_OLW_TYPE)

# print('Validating...')
# validate(automaton, load_seqs_from_file(VALIDATION_FILE))

