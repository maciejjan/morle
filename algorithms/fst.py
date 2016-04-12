import libhfst
import settings
import sys

def seq_to_transducer(alignment, weight=0.0, type=settings.TRANSDUCER_TYPE, alphabet=None):
	tr = libhfst.HfstBasicTransducer()
	if alphabet is not None:
		tr.add_symbols_to_alphabet(alphabet)
	last_state_id = 0
	for (x, y) in alignment:
		state_id = tr.add_state()
		tr.add_transition(state_id, libhfst.HfstBasicTransition(state_id, libhfst.EPSILON, libhfst.EPSILON, 0.0))
		if (x, y) == (libhfst.IDENTITY, libhfst.IDENTITY):
			tr.add_transition(last_state_id, 
							  libhfst.HfstBasicTransition(state_id,
														  libhfst.IDENTITY,
														  libhfst.IDENTITY,
														  0.0))
			tr.add_transition(state_id, 
							  libhfst.HfstBasicTransition(state_id,
														  libhfst.IDENTITY,
														  libhfst.IDENTITY,
														  0.0))
			for a in tr.get_alphabet():
				if not a.startswith('@_'):
					tr.add_transition(last_state_id, libhfst.HfstBasicTransition(state_id, a, a, 0.0))
					tr.add_transition(state_id, libhfst.HfstBasicTransition(state_id, a, a, 0.0))
		else:
			tr.add_transition(last_state_id, 
							  libhfst.HfstBasicTransition(state_id, x, y, 0.0))
		last_state_id = state_id
	tr.set_final_weight(last_state_id, weight)
	return libhfst.HfstTransducer(tr, type)

def binary_disjunct(transducers):
	stack, sizes = [], []
#	print('starting')
	count = 0
	while True:
		if len(sizes) >= 2 and sizes[-1] == sizes[-2]:
			first, first_size = stack.pop(), sizes.pop()
			second, second_size = stack.pop(), sizes.pop()
			first.disjunct(second)
			stack.append(first)
			sizes.append(first_size + second_size)
#			stack[-1].minimize()
		else:
#			print('expand')
			try:
				stack.append(next(transducers))
				sizes.append(1)
				count += 1
				sys.stdout.write('\r'+str(count))
			except StopIteration:
				break
	print()
#	print('final disjunction')
	t = stack.pop()
	while stack:
		t.disjunct(stack.pop())
#	print('postprocessing')
	t.minimize()
	t.push_weights(libhfst.TO_INITIAL_STATE)
#	print('adding epsilon loops and word boundaries')
	t = libhfst.HfstBasicTransducer(t)
#	add_epsilon_loops(t)
#	add_word_boundaries(t)
	return libhfst.HfstTransducer(t, settings.TRANSDUCER_TYPE)

