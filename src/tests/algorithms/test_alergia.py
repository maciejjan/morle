from morle.algorithms.alergia import *
import morle.shared as shared

import math
import random
import unittest


class FrequencyAutomatonStateTest(unittest.TestCase):

    def test_all(self) -> None:
        state_id = random.randrange(100000)
        state = FrequencyAutomatonState(state_id)
        self.assertEqual(state.id, state_id)
        self.assertEqual(state.final_freq, 0)
        self.assertEqual(state.get_total_freq(), 0)


class FrequencyAutomatonTransitionTest(unittest.TestCase):

    def test_all(self) -> None:
        source_state_id = random.randrange(100000)
        target_state_id = random.randrange(100000)
        symbol = 'X'
        tr = FrequencyAutomatonTransition(source_state_id, symbol,
                                          target_state_id)
        self.assertEqual(tr.source_state_id, source_state_id)
        self.assertEqual(tr.target_state_id, target_state_id)


class FrequencyAutomatonTest(unittest.TestCase):

    def test_add_and_delete_state(self) -> None:
        automaton = FrequencyAutomaton()
        new_state = automaton.add_state()
        tr = automaton.add_transition(0, 'c', new_state)
        del automaton[new_state]
        # test whether the transition has also been deleted
        with self.assertRaises(KeyError):
            automaton[0]['c'].increase_freq(1)

    def test_add_and_delete_transition(self) -> None:
        automaton = FrequencyAutomaton()
        new_state = automaton.add_state()
        tr = automaton.add_transition(0, 'c', new_state)
        self.assertEqual(len(automaton[0].ingoing_transitions), 0)
        self.assertEqual(len(automaton[0].transitions), 1)
        self.assertEqual(len(automaton[1].ingoing_transitions), 1)
        self.assertEqual(len(automaton[1].transitions), 0)
        self.assertEqual(tr, automaton[0].transitions['c'])
        self.assertIn(tr, automaton[1].ingoing_transitions)

        automaton.delete_transition(tr)
        self.assertEqual(len(automaton[0].transitions), 0)
        self.assertEqual(len(automaton[1].ingoing_transitions), 0)

    def test_merge_states(self) -> None:
        # build the following automaton:
        #      ,---a(18)---> 1(6) ---b(9)---> 3(2) ---c(7)---> 7(7)
        # 0(0)                   `---c(3)---> 4(3)
        #      \
        #       `---b(6)---> 2(0) ---b(4)---> 5(3) ---c(1)---> 8(1)
        #                        `---c(2)---> 6(2)
        automaton = FrequencyAutomaton()
        for i in range(8):
            automaton.add_state()
        automaton.add_transition(0, 'a', 1, 18)
        automaton.add_transition(0, 'b', 2, 6)
        automaton.add_transition(1, 'b', 3, 9)
        automaton.add_transition(1, 'c', 4, 3)
        automaton.add_transition(2, 'b', 5, 4)
        automaton.add_transition(2, 'c', 6, 2)
        automaton.add_transition(3, 'c', 7, 7)
        automaton.add_transition(5, 'c', 8, 1)
        automaton[0].final_freq = 0
        automaton[1].final_freq = 6
        automaton[2].final_freq = 0
        automaton[3].final_freq = 2
        automaton[4].final_freq = 3
        automaton[5].final_freq = 3
        automaton[6].final_freq = 2
        automaton[7].final_freq = 7
        automaton[8].final_freq = 1

        automaton.merge_states(0, 1)

        # expected result:
        #    .a(18).
        #    \   /
        #     \ V
        #     0(6) ---b(15)---> 2(2) ---b(4)---> 5(3) ---c(1)---> 8(1)
        #      \                    `---c(9)---> 6(9)
        #       `---c(3)---> 4(3)
        
        # test frequencies in the root state
        self.assertEqual(automaton[0].final_freq, 6)
        self.assertEqual(automaton[0]['a'].freq, 18)
        self.assertEqual(automaton[0]['b'].freq, 15)
        self.assertEqual(automaton[0]['c'].freq, 3)
        
        # test frequencies in the state reach from root by symbol 'b'
        state = automaton[0]['b'].target_state_id
        self.assertEqual(automaton[state].final_freq, 2)
        self.assertEqual(automaton[state]['b'].freq, 4)
        self.assertEqual(automaton[state]['c'].freq, 9)

        # test state numbering
        self.assertEqual(automaton[0]['a'].target_state_id, 0)
        self.assertEqual(automaton[0]['b'].target_state_id, 2)
        self.assertEqual(automaton[0]['c'].target_state_id, 4)
        self.assertEqual(automaton[2]['b'].target_state_id, 5)
        self.assertEqual(automaton[2]['c'].target_state_id, 6)
        self.assertEqual(automaton[5]['c'].target_state_id, 8)

        # test final frequencies
        self.assertEqual(automaton[0].final_freq, 6)
        self.assertEqual(automaton[2].final_freq, 2)
        self.assertEqual(automaton[4].final_freq, 3)
        self.assertEqual(automaton[5].final_freq, 3)
        self.assertEqual(automaton[6].final_freq, 9)
        self.assertEqual(automaton[8].final_freq, 1)

        # test ingoing transitions
        self.assertEqual(
            automaton[0].ingoing_transitions[0].source_state_id, 0)
        self.assertEqual(
            automaton[2].ingoing_transitions[0].source_state_id, 0)
        self.assertEqual(
            automaton[4].ingoing_transitions[0].source_state_id, 0)
        self.assertEqual(
            automaton[5].ingoing_transitions[0].source_state_id, 2)
        self.assertEqual(
            automaton[6].ingoing_transitions[0].source_state_id, 2)
        self.assertEqual(
            automaton[8].ingoing_transitions[0].source_state_id, 5)

        # test deleted states
        for state in [1, 3, 7]:
            with self.assertRaises(KeyError):
                automaton[state]

    def test_to_hfst(self) -> None:
        # test using this simple automaton:
        #  ,---a(15)---> 1(15)
        # 0
        #  `---b(10)---> 2(10)
        automaton = FrequencyAutomaton()
        for i in range(2):
            automaton.add_state()
        automaton.add_transition(0, 'a', 1)
        automaton.add_transition(0, 'b', 2)
        automaton[0]['a'].increase_freq(10)
        automaton[automaton[0]['a'].target_state_id].increase_final_freq(10)
        automaton[0]['b'].increase_freq(10)
        automaton[automaton[0]['b'].target_state_id].increase_final_freq(10)
        automaton[0]['a'].increase_freq(5)
        automaton[automaton[0]['a'].target_state_id].increase_final_freq(5)

        hfst_automaton = automaton.to_hfst()
        self.assertIsInstance(hfst_automaton, hfst.HfstTransducer)
        self.assertEqual(hfst_automaton.get_type(),
                         hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        hfst_basic_automaton = hfst.HfstBasicTransducer(hfst_automaton)
        for transition in hfst_basic_automaton.transitions(0):
            expected_weight = None
            if transition.get_input_symbol() == 'a':
                expected_weight = -math.log(0.6)
            elif transition.get_input_symbol() == 'b':
                expected_weight = -math.log(0.4)
            self.assertAlmostEqual(transition.get_weight(), expected_weight)
            self.assertAlmostEqual(
                hfst_basic_automaton.get_final_weight(
                    transition.get_target_state()),
                0.0)

    def test_topological_ordering(self) -> None:
        automaton = FrequencyAutomaton()
        for i in range(6):
            automaton.add_state()

        # create the following automaton:
        #  ,-a-> 1 -b-> 2 -c-> 3
        # 0
        #  `-b-> 4 -b-> 5 -c-> 6
        automaton.add_transition(0, 'a', 1)
        automaton.add_transition(1, 'b', 2)
        automaton.add_transition(2, 'c', 3)
        automaton.add_transition(0, 'b', 4)
        automaton.add_transition(4, 'b', 5)
        automaton.add_transition(5, 'c', 6)
        automaton[3].increase_final_freq(1)
        automaton[6].increase_final_freq(2)
        automaton.rename_states_to_topological_ordering()

        # expected result:
        #  ,-a-> 1 -b-> 3 -c-> 5
        # 0
        #  `-b-> 2 -b-> 4 -c-> 6
        self.assertEqual(automaton[0]['a'].target_state_id, 1)
        self.assertEqual(automaton[0]['b'].target_state_id, 2)
        self.assertEqual(automaton[1]['b'].target_state_id, 3)
        self.assertEqual(automaton[2]['b'].target_state_id, 4)
        self.assertEqual(automaton[3]['c'].target_state_id, 5)
        self.assertEqual(automaton[4]['c'].target_state_id, 6)
        self.assertEqual(automaton[5].final_freq, 1)
        self.assertEqual(automaton[6].final_freq, 2)

        # now introduce an unreachable state
        automaton.add_state()
        with self.assertRaises(ValueError):
            automaton.rename_states_to_topological_ordering()


class PrefixTreeAcceptorTest(unittest.TestCase):

    def setUp(self):
        self.seqs_with_freqs = [\
            (['e', 'x', 'a', 'm', 'p', 'l', 'e'], 1),
            (['e', 'x', 'p', 'l', 'o', 's', 'i', 'o', 'n'], 3),
            (['{CAP}', 'p', 'a', 'r', 'i', 's'], 2),
            (['{CAP}', 'p', 'a', 'r', 'i', 's', 'h'], 7)
        ]
        self.seqs = []
        for seq, freq in self.seqs_with_freqs:
            for i in range(freq):
                self.seqs.append(seq)
        self.pta = prefix_tree_acceptor(self.seqs)

    def test_lookup(self):
        for seq, freq in self.seqs_with_freqs:
            cur_state = 0
            for c in seq:
                cur_state = self.pta[cur_state][c].target_state_id
            self.assertEqual(self.pta[cur_state].final_freq, freq)

    def test_structure(self):
        self.assertEqual(len(self.pta.states), 22)
        self.assertEqual(self.pta[self.pta.initial_state_id].get_total_freq(),
                         len(self.seqs))


class AlergiaAlgorithmTest(unittest.TestCase):

    def setUp(self):
        # learn the language: (a|c)b+c
        self.seqs = [\
            ['a', 'b', 'c'],
            ['a', 'b', 'b', 'c'],
            ['a', 'b', 'b', 'b', 'c'],
            ['c', 'b', 'c'],
            ['c', 'b', 'b', 'c'],
            ['c', 'b', 'b', 'b', 'c']
        ]
    
    def test_alergia(self):
        pass # TODO
#         automaton = alergia(self.seqs)
        
# TODO test with various frequency thresholds

