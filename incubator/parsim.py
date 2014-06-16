import algorithms.clustering
import algorithms.cooccurrences
import algorithms.vectorsim
from datastruct.graph import *
import datastruct.storage
from utils.files import *
from utils.printer import *
import utils.db

MAX_COOC = 50

# TODO error: clusters 17 and 31 in pol_lei_300k should be similar

def filter_par_cooc(filename, max_cooc):
	outfile_name = filename + '.filtered'
	with open_to_write(outfile_name) as outfp:
		for p_id, coocs in load_tsv_file_by_key(filename, 1):
			coocs.sort(reverse = True, key = lambda x: float(x[1]))
			for p2_id, sig in coocs[:max_cooc]:
				write_line(outfp, (p_id, p2_id, sig))
#	rename_file(outfile_name, filename)

def content_sim():
	algorithms.vectorsim.calculate_vector_sim('par_rul.txt', 'par_cos_sim.txt')

def context_sim():
	algorithms.vectorsim.calculate_vector_sim('par_co_n.txt', 'par_con_sim.txt')

def cooc_sim():
	algorithms.cooccurrences.calculate_cooc('par_co_n.txt', 'par_co_2.txt')

def cluster_paradigms(input_file, output_file, v1_col=1, v2_col=2, sim_col=3):
#	graph = Graph(storage=datastruct.storage.TRIE)
	graph = Graph()
	pp = progress_printer(get_file_size(input_file))
	for row in load_tsv_file(input_file):
		graph.add_vertex(row[v1_col-1])
		graph.add_vertex(row[v2_col-1])
		graph.add_edge(row[v1_col-1], row[v2_col-1], float(row[sim_col-1]))
		pp.next()
	clusters = algorithms.clustering.clink(graph)
#	clusters = algorithms.clustering.chinwhisp(graph)
	with open_to_write(output_file) as outfp:
		for cl_id, cl in enumerate(clusters, 1):
			for par_id in cl:
				write_line(outfp, (par_id, cl_id))

def join_tables():
	keys = set([])
	par_cos_sim = {}
	for row in load_tsv_file('par_cos_sim.txt'):
		par_cos_sim[(row[0], row[1])] = float(row[2])
		keys.add((row[0], row[1]))
	par_con_sim = {}
	for row in load_tsv_file('par_con_sim.txt'):
		par_con_sim[(row[0], row[1])] = float(row[2])
		keys.add((row[0], row[1]))
	par_co_2 = {}
	for row in load_tsv_file('par_co_2.txt'):
		par_co_2[(row[0], row[1])] = float(row[3])
		keys.add((row[0], row[1]))
	with open_to_write('par_sim.txt') as fp:
		for k in keys:
			cos = par_cos_sim[k] if par_cos_sim.has_key(k) else 0.0
			con = par_con_sim[k] if par_con_sim.has_key(k) else 0.0
			co_2 = par_co_2[k] if par_co_2.has_key(k) else 0.0
			write_line(fp, (k[0], k[1], cos, con, co_2))

def run():
#	content_sim()
#	filter_par_cooc('par_co_n.txt', MAX_COOC)
#	context_sim()
#	cooc_sim()
#	join_tables()
#	print 'Filtering paradigm similarities...'
#	filter_par_cooc('par_cos_sim.txt', MAX_COOC)
	print 'Clustering paradigms...'
	cluster_paradigms('par_cos_sim.txt.filtered', 'par_cos_cl.txt')

def export_to_db():
	settings.DB_HOST = 'localhost'
	settings.DB_USER = 'ja'
	settings.DB_PASS = 'DhvRf1N!'
	settings.DB_NAME = 'pol_lei_300k'
	utils.db.connect()
	utils.db.query('ALTER TABLE paradigms ADD cluster INT DEFAULT NULL;')
	for p_id, cluster in load_tsv_file('par_cos_cl.txt'):
		utils.db.query('UPDATE paradigms SET cluster = %s WHERE p_id = %s' % (cluster, p_id))
	utils.db.close_connection()
