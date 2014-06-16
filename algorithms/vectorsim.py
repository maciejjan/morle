from utils.files import *
from utils.printer import *
import math

### SIMILARITY FUNCTIONS ###

def cosine(vec_1, vec_2):
	def weight(elem):
		return weights[elem] if weights is not None else 1.0
	v1, v2 = [], []
	set_1 = set(vec_1.keys())
	set_2 = set(vec_2.keys())
	for elem in (set_1 & set_2):
		v1.append(vec_1[elem])
		v2.append(vec_2[elem])
	for elem in (set_1 - set_2):
		v1.append(vec_1[elem])
		v2.append(0.0)
	for elem in (set_2 - set_1):
		v1.append(0.0)
		v2.append(vec_2[elem])
	prod = sum([x*y for x, y in zip(v1, v2)])
	v1_len = math.sqrt(sum([x*x for x in v1]))
	v2_len = math.sqrt(sum([x*x for x in v2]))
	return prod / (v1_len * v2_len)

#def dice(set_1, set_2):
#	def weight(elem):
#		return weights[elem] if weights is not None else 1.0
#	def norm(set_):
#		return sum([weight(e) for e in set_])
#	return 2*norm(set_1 & set_2) / (norm(set_1) + norm(set_2))

SIM_FUN = cosine

### MAIN FUNCTION ###

def calculate_vector_sim(input_file, output_file):
	vectors_file = 'vectors.txt'		# TODO change
	# create vectors file: vec_id<tab>elem<tab>all_elems
	pp = progress_printer(get_file_size(input_file))
	with open_to_write(vectors_file) as vec_fp:
		for key, values in load_tsv_file_by_key(input_file, 1):
			values_str = '  '.join(sorted([v[0]+' '+v[1] for v in values]))
			for (v, w) in values:
				write_line(vec_fp, (key, v, values_str))
			for i in range(len(values)):
				pp.next()
	sort_file(vectors_file, key=2)
	# for each pair of vectors with common element:
	#   if the common element is minimal => calculate similarity
	pp = progress_printer(get_file_size(vectors_file))
	with open_to_write(output_file) as out_fp:
		for key, values in load_tsv_file_by_key(vectors_file, 2):
			for (vec_id_1, vec_elems_1_str) in values:
				for (vec_id_2, vec_elems_2_str) in values:
					if vec_id_1 < vec_id_2:
						vec_elems_1, vec_elems_1_keys = {}, set([])
						for el in vec_elems_1_str.split('  '):
							el = el.split(' ')
							vec_elems_1[el[0]] = float(el[1])
							vec_elems_1_keys.add(el[0])
						vec_elems_2, vec_elems_2_keys = {}, set([])
						for el in vec_elems_2_str.split('  '):
							el = el.split(' ')
							vec_elems_2[el[0]] = float(el[1])
							vec_elems_2_keys.add(el[0])
						if min(vec_elems_1_keys & vec_elems_2_keys) != key:
							continue
						if len(vec_elems_1_keys & vec_elems_2_keys) < \
							max(len(vec_elems_1_keys), len(vec_elems_2_keys)) / 2: 
							continue
						write_line(out_fp, (vec_id_1, vec_id_2, \
							SIM_FUN(vec_elems_1, vec_elems_2)))
			for i in range(len(values)):
				pp.next()

