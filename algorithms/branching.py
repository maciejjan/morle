# find maximum branching

def branching(vertices, edges):

	def add_weight(queue, value):
		new_queue = [(v1, v2, rule, weight + value) for (v1, v2, rule, weight) in queue]
		return new_queue

	val = min([e[3] for e in edges])
	edges = add_weight(edges, -val + 0.1)
	roots = list(vertices)
	entering_edges, enter, strong, strong_sets, weak, weak_sets, root = {}, {}, {}, {}, {}, {}, {}
	for v in vertices:
		entering_edges[v] = []
		strong[v] = v
		strong_sets[v] = [v]
		weak[v] = v
		weak_sets[v] = [v]
		enter[v] = None
		root[v] = v
	for (v1, v2, rule, weight) in edges:
		entering_edges[v2].append((v1, v2, rule, weight))
	for v in vertices:
		entering_edges[v].sort(key=lambda x: x[3])
#		print v, entering_edges[v]
	result = []
	final = []

	while roots:
		k = roots.pop()
#		print k
		if not entering_edges[k]:
			final.append(k)
			continue
		v1, v2, rule, weight = entering_edges[k].pop()
#		print v1, v2, strong[v1], strong[v2], rule, weight
		if strong[v1] == k:
			roots.append(k)
		else:
			result.append((v1, v2, rule))
			if weak[v1] != weak[v2]:
#				w[v1].extend(w[v2])
#				print '  merging weak sets:', v1, v2
				weak_sets[weak[v1]].extend(weak_sets[weak[v2]])
				to_delete = weak[v2]
				for v in weak_sets[to_delete]:
					weak[v] = weak[v1]
				del weak_sets[to_delete]
				enter[k] = (v1, v2, rule, weight)
			else:
				x, y, r, w = v1, v2, rule, weight
				val, vertex = None, None
				while True:
					if val is None or w < val:
						val = w
						vertex = strong[y]
					if enter[strong[x]]:
						x, y, r, w = enter[strong[x]]
					else:
						break
				entering_edges[k] = add_weight(entering_edges[k], val - weight)
				root[k] = root[vertex]
				x, y, r, w = enter[strong[v1]]
				while True:
					entering_edges[strong[y]] = add_weight(entering_edges[strong[y]], val - w)
					entering_edges[k].extend(entering_edges[strong[y]])
					entering_edges[k].sort(key=lambda x: x[3])
					del entering_edges[strong[y]]
					try:
						roots.remove(y)
					except Exception:
						pass
					strong_sets[k].extend(strong_sets[strong[y]])
#					k.extend(s[y])
#					print '  merging strong sets:', k, strong[y]
					to_delete = strong[y]
					for yy in strong_sets[to_delete]:
						strong[yy] = k
					del strong_sets[to_delete]
#					strong[y] = k
					if enter[strong[x]]:
						x, y, r, w = enter[strong[x]]
					else:
						break
				roots.append(k)

	edges_final = []
	root_vertices = set([root[x] for x in final])
	print 'ROOT' in root_vertices
	while result:
		v1, v2, rule = result.pop(0)
		if v1 in root_vertices:
			if not v2 in root_vertices:
				edges_final.append((v1, v2, rule))
				root_vertices.add(v2)
		else:
			result.append((v1, v2, rule))

	return edges_final
	
	return result, root, final

