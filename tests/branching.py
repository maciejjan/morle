import sys
sys.path.insert(0, '../../src/')
import algorithms.branching

branching, root, final = algorithms.branching.branching(['a', 'b', 'c'],
	[('a', 'b', '', 0.1),\
	('b', 'c', '', 0.2),\
	('c', 'b', '', 0.1),\
	('b', 'a', '', 0.3),\
	('a', 'c', '', 0.5)])

