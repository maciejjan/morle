'''ALERGIA algorithm for learning deterministic automata.'''

from collections import defaultdict
import hfst
import math
from operator import itemgetter
import re
import sys
from typing import Iterable


class FrequencyAutomatonState:
    def __init__(self, state_id :int) -> None:
        self.id = state_id
        self.ingoing_transitions = []
        self.transitions = {}
        self.final_freq = 0

    def __contains__(self, key :str) -> bool:
        return key in self.transitions

    def __getitem__(self, key :str):
        return self.transitions[key]

    def __iter__(self):
        return iter(sorted(self.transitions.values(),
                           key=lambda tr: tr.symbol))

    def increase_final_freq(self, value :int) -> None:
        self.final_freq += value

    def get_total_freq(self) -> int:
        return self.final_freq +\
               sum(tr.freq for tr in self.transitions.values())


class FrequencyAutomatonTransition:
    def __init__(self, source_state_id :int, symbol :str,  
                 target_state_id :int, freq :int = 0) -> None:
        self.source_state_id = source_state_id
        self.symbol = symbol
        self.target_state_id = target_state_id
        self.freq = freq

    def increase_freq(self, value :int) -> None:
        self.freq += value


class FrequencyAutomaton:
    def __init__(self) -> None:
        self.states = {}            # type: Dict[int, FrequencyAutomatonState]
        self.next_state_id = 0
        self.initial_state_id = self.add_state()

    def __getitem__(self, key :int) -> FrequencyAutomatonState:
        return self.states[key]

    def __setitem__(self, key :int, val :FrequencyAutomatonState) -> None:
        self.states[key] = val

    def __delitem__(self, key :int) -> None:
        for transition in list(self.states[key].ingoing_transitions):
            self.delete_transition(transition)
        del self.states[key]

    def add_state(self) -> int:
        added_state_id = self.next_state_id
        self[added_state_id] = FrequencyAutomatonState(added_state_id)
        self.next_state_id += 1
        return added_state_id

    def add_transition(self, source_state_id :int, symbol :str,
                       target_state_id :int, freq :int = 0) \
                       -> FrequencyAutomatonTransition:
        tr = FrequencyAutomatonTransition(source_state_id, symbol, 
                                          target_state_id, freq)
        self[source_state_id].transitions[symbol] = tr
        self[target_state_id].ingoing_transitions.append(tr)
        return tr

    def delete_transition(self, transition :FrequencyAutomatonTransition) \
                         -> None:
        self[transition.target_state_id].ingoing_transitions.remove(transition)
        del self[transition.source_state_id].transitions[transition.symbol]

    def fold_states(self, state_1_id :int, state_2_id :int) -> None:
        self[state_1_id].increase_final_freq(self[state_2_id].final_freq)
        for tr_2 in self[state_2_id]:
            self.delete_transition(tr_2)
            if tr_2.symbol in self[state_1_id]:
                tr_1 = self[state_1_id][tr_2.symbol]
                tr_1.increase_freq(tr_2.freq)
                self.fold_states(tr_1.target_state_id, tr_2.target_state_id)
            else:
                self[tr_2.target_state_id].ingoing_transitions = []
                self.add_transition(state_1_id, tr_2.symbol, 
                                    tr_2.target_state_id, tr_2.freq)
        del self[state_2_id]

    def merge_states(self, state_1_id :int, state_2_id :int) -> None:
        if len(self[state_2_id].ingoing_transitions) > 1:
            raise ValueError('Attempting to merge a state with more than one'
                             'ingoing transition!')
        elif len(self[state_2_id].ingoing_transitions) == 0:
            raise ValueError('Attempting to merge the root state!')
        parent_tr = self[state_2_id].ingoing_transitions[0]
        # delete parent_tr
        self.delete_transition(parent_tr)
        # add the counterpart of parent_tr leading to state_1
        self.add_transition(parent_tr.source_state_id, parent_tr.symbol,
                            state_1_id, parent_tr.freq)
        self.fold_states(state_1_id, state_2_id)

    def rename_states_to_topological_ordering(self):
        '''Rename states, so that their IDs are in topological order.'''
        # determine the new state IDs, which are topologically sorted
        queue = [self.initial_state_id]
        id_mapping = {} # type: Dict[int, int]
        cur_mapped_id = 0
        while queue:
            state_id = queue.pop(0)
            id_mapping[state_id] = cur_mapped_id
            cur_mapped_id += 1
            queue.extend(tr.target_state_id for tr in self[state_id])
        # replace the state IDs with new ones
        new_states = {}
        for state_id, state in self.states.items():
            try:
                state.id = id_mapping[state_id]
                new_states[id_mapping[state_id]] = state
                for tr in state:
                    tr.source_state_id = id_mapping[tr.source_state_id]
                    tr.target_state_id = id_mapping[tr.target_state_id]
            except KeyError:
                raise ValueError('Unreachable state: %d' % state_id)
        self.states = new_states

    def to_hfst(self) -> hfst.HfstTransducer:
        result = hfst.HfstBasicTransducer()
        for state in self.states.values():
            total_freq = state.get_total_freq()
            for t in state.transitions.values():
                weight = -math.log(t.freq / total_freq)
                result.add_transition(state.id,
                    hfst.HfstBasicTransition(t.target_state_id,
                                             t.symbol, t.symbol,
                                             weight))
            if state.final_freq > 0:
                final_weight = -math.log(state.final_freq / total_freq)
                result.set_final_weight(state.id, final_weight)
        return hfst.HfstTransducer(result)


def prefix_tree_acceptor(seqs :Iterable[Iterable[str]]) -> FrequencyAutomaton:

    def _pta_insert(automaton, seq, freq=1, state=0):
        if not seq:
            automaton[state].increase_final_freq(freq)
        else:
            if seq[0] in automaton[state]:
                next_tr = automaton[state][seq[0]]
            else:
                next_state = automaton.add_state()
                next_tr = automaton.add_transition(state, seq[0], next_state)
            next_tr.increase_freq(freq)
            _pta_insert(automaton, seq[1:], freq, next_tr.target_state_id)

    automaton = FrequencyAutomaton()
    for i, seq in enumerate(seqs):
        _pta_insert(automaton, seq, freq=1)
    return automaton


def alergia(seqs :Iterable[Iterable[str]], 
            alpha :float = 0.05, 
            freq_threshold :int = 1) -> FrequencyAutomaton:

    automaton = prefix_tree_acceptor(seqs)
    automaton.rename_states_to_topological_ordering()

    def _test(f1 :int, n1 :int, f2 :int, n2 :int) -> bool:
        diff = abs(f1/n1 - f2/n2)
        threshold = (1/math.sqrt(n1) + 1/math.sqrt(n2)) *\
                    math.sqrt(0.5*math.log(2/alpha))
        return diff < threshold

    def _compatible(state_1_id :int, state_2_id :int) -> bool:
        n1 = automaton[state_1_id].get_total_freq()
        n2 = automaton[state_2_id].get_total_freq()
        for key in set(automaton[state_1_id].transitions.keys()) |\
                   set(automaton[state_2_id].transitions.keys()):
            f1 = automaton[state_1_id].transitions[key].freq \
                 if key in automaton[state_1_id] else 0
            f2 = automaton[state_2_id].transitions[key].freq \
                 if key in automaton[state_2_id] else 0
            if not _test(f1, n1, f2, n2):
                return False
        return True

    red_states = {0}
    blue_states = { tr.target_state_id \
                    for tr in automaton[0].transitions.values() }

    while blue_states:
        q_b = min(blue_states)
        blue_states.remove(q_b)
        merged = False
        for q_r in red_states:
            if _compatible(q_r, q_b):
                automaton.merge_states(q_r, q_b)
                merged = True
                break
        if not merged:
            red_states.add(q_b)
        blue_states = { tr.target_state_id \
                        for q in red_states for tr in automaton[q]\
                        if automaton[tr.target_state_id].get_total_freq() >\
                           freq_threshold
                      } -\
                      red_states

    return automaton

