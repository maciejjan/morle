# find maximum branching
from utils.files import open_to_write, write_line
#import heapq
import random

def branching(vertices, edges):
	roots = list(vertices)
	random.shuffle(roots)
	entering_edges, enter, min = {}, {}, {}
	strong, strong_sets = {}, {}
	weak, weak_sets = {}, {}
	edge_weights = {}

	def add_weight_to_queue(queue, value):
		for e in queue:
			edge_weights[e] += value
	
	def merge_weak_sets(weak_set_1, weak_set_2):
#		print('merging weak sets:', weak_set_1.key, weak_set_2.key)
		weak_sets[weak_set_1].extend(weak_sets[weak_set_2])
#		to_delete = weak_set_2
		for v in weak_sets[weak_set_2]:
			weak[v] = weak_set_1
#		del weak_sets[to_delete]
		del weak_sets[weak_set_2]
	
	def merge_strong_sets(strong_set_1, strong_set_2):
#		print('merging strong sets:', strong_set_1.key, strong_set_2.key)
		strong_sets[strong_set_1].extend(strong_sets[strong_set_2])
#		to_delete = strong[key_2]
		for v in strong_sets[strong_set_2]:
			strong[v] = strong_set_1
#		del strong_sets[to_delete]
		del strong_sets[strong_set_2]
	
	for v in vertices:
		entering_edges[v] = []
		strong[v] = v
		strong_sets[v] = [v]
		weak[v] = v
		weak_sets[v] = [v]
		enter[v] = None
		min[v] = v
	for e in edges:
		edge_weights[e] = e.target.cost - e.cost
		entering_edges[e.target].append(e)
	for queue in entering_edges.values():
		queue.sort(key=lambda e: edge_weights[e])
	result = []
	rset = []

	while roots:
		k = roots.pop()
#		print('k =', k.key)
		if not entering_edges[k] or edge_weights[entering_edges[k][-1]] <= 0.0:
			rset.append(k)
			continue
		e = entering_edges[k].pop()
#		if edge_weights[e] <= 0.0:
#			print([edge_weights[x] for x in entering_edges[k]])
#			raise Exception('!')
#		print(e.source.key, e.target.key, strong[e.source].key, strong[e.target].key, str(e.rule), edge_weights[e])
		if strong[e.source] == k:
			roots.append(k)
#			print('strong sets equal')
		else:
			result.append(e)
			if weak[e.source] != weak[e.target]:
#				print('weak sets different -> merging')
				merge_weak_sets(weak[e.source], weak[e.target])
				enter[k] = e
			else:
#				print('merging to one strong set')
				e2 = e
				val, vertex = None, None
				while True:
					if val is None or edge_weights[e2] < val:
						val = edge_weights[e2]
						vertex = strong[e2.target]
					if enter[strong[e2.source]]:
						e2 = enter[strong[e2.source]]
					else:
						break
				add_weight_to_queue(entering_edges[k], val-edge_weights[e])
				min[k] = min[vertex]
				e2 = enter[strong[e.source]]
				while True:
					add_weight_to_queue(\
						entering_edges[strong[e2.target]],\
						val-edge_weights[e2])
					# qunion(k, sfind(y))
					entering_edges[k].extend(\
						entering_edges[strong[e2.target]])
					entering_edges[k].sort(key=lambda x: edge_weights[x])
					del entering_edges[strong[e2.target]]
					# sunion(k, sfind(y))
#					try:
#						roots.remove(e2.target)
#					except Exception:
#						pass
					merge_strong_sets(k, strong[e2.target])
					if enter[strong[e2.source]]:
						e2 = enter[strong[e2.source]]
					else:
						break
				roots.append(k)
	
	print('result cost = %f' % sum(e.cost for e in result))
	with open_to_write('result.txt') as fp:
		for e in result:
			write_line(fp, (e.source.key, e.target.key))

	with open_to_write('strong.txt') as fp:
		for ss in strong_sets.values():
			write_line(fp, (', '.join(sorted([n.key for n in ss])), ))
#	for ss in strong_sets.values():
#		print(', '.join(sorted([n.key for n in ss])))

	edges_final = []
	root_vertices = set([min[x] for x in rset])
#	print 'ROOT' in root_vertices
	with open_to_write('extract.txt') as fp:
		while result:
			e = result.pop(0)
			if e.source in root_vertices:
				if not e.target in root_vertices:
					edges_final.append(e)
					root_vertices.add(e.target)
					fp.write('adding edge: %s -> %s\n' % (e.source.key, e.target.key))
				else:
					fp.write('discarding edge: %s -> %s\n' % (e.source.key, e.target.key))
			else:
				result.append(e)

	return edges_final
	
#	return result, root, final

