from utils.files import *
from storage import *

class Graph:
	def __init__(self, storage=DICT, edges_storage=LIST, directed=False):
		self.directed = directed
		self.data = new_storage_struct(storage)
		self.edges_storage = edges_storage
	
	def add_edge(self, v1, v2, w=1):
		if not self.has_edge(v1, v2):
			self.data[v1][v2] = Edge(v2, w)
	
	def add_vertex(self, v):
		if not self.has_vertex(v):
			self.data[v] = new_storage_struct(self.edges_storage)
	
#	def delete_vertex(self, v):
#		del self.data[v]
		# TODO delete all edges - or throw away this function?
	
#	def delete_edge(self, v1, v2):
#		# TODO
#		pass
	
	def edges(self, v):
		return self.data[v].values()
	
	def get_edge_weight(self, v1, v2):
		return self.data[v1][v2].weight
	
	def has_edge(self, v1, v2):
		if not self.has_vertex(v1):
			raise KeyError(v1)
		if not self.has_vertex(v2):
			raise KeyError(v2)
		return self.data[v1].has_key(v2)
	
	def has_vertex(self, v):
		return self.data.has_key(v)

	@staticmethod
	def load_from_file(filename, v1_col=0, v2_col=1, w_col=2):
		def load_col(row, cols):
			if isinstance(cols, int):
				return row[cols]
			else:
				return tuple([row[c] for c in cols])
		graph = Graph()
		for row in load_tsv_file(filename):
			v1 = load_col(row, v1_col)
			v2 = load_col(row, v2_col)
			w = None
			if isinstance(w_col, int):
				w = float(row[w_col])
			elif hasattr(w_col, '__call__'):
				w = w_col(row)
			if not graph.has_vertex(v1):
				graph.add_vertex(v1)
			if not graph.has_vertex(v2):
				graph.add_vertex(v2)
			if w is not None:
				graph.add_edge(v1, v2, w)
			else:
				graph.add_edge(v1, v2)
		return graph
	
	def save_to_file(self, filename):
		def vertex_to_str(v):
			if isinstance(v, tuple):
				return '\t'.join([unicode(x) for x in v])
			else:
				return v
		lines_written = 0
		with open_to_write(filename) as fp:
			for v in self.vertices():
				for e in self.edges(v):
					v1_str = vertex_to_str(v)
					v2_str = vertex_to_str(e.vertex)
					write_line(fp, (v1_str, v2_str, e.weight))
					lines_written += 1
		set_file_size(filename, lines_written)

	def vertices(self):
		return self.data.keys()

class Edge:
	def __init__(self, vertex, weight=1):
		self.vertex = vertex
		self.weight = weight

	def __eq__(self, other):
		return self.vertex == other.vertex

