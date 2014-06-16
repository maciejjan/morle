import algorithms.align
import algorithms.clustering
from datastruct.counter import *
from datastruct.graph import *
from datastruct.rules import *
import datastruct.storage
import utils.db
from utils.files import *
from utils.printer import *
import settings
import math

### LEARNING LEXEMES ###

def cluster_rules(s_rul_file, s_rul_cooc_file):
	# load edges as graph
	edges_g = Graph.load_from_file(s_rul_cooc_file, 0, 1, 3)
	for r, freq in load_tsv_file(s_rul_file):	# add isolated vertices (TODO: do not load the file twice!)
		edges_g.add_vertex(r)
	# cluster
	clusters = algorithms.clustering.clink(edges_g)
#	clusters = algorithms.clustering.chinwhisp(edges_g, max_iterations=20)
	# return the clusters?
	return clusters

def entropy(rules_c, total):
	def compute_entr(rule):
		p = float(rules_c[rule]) / total
		if p == 0:
			return 0
		return -p * math.log(p) - (1-p) * math.log(1-p)
	return compute_entr

def logfreq(rules_c):
	def compute_logfreq(rule):
		return 0 if not rules_c.has_key(rule) \
			else math.log(rules_c[rule]) - math.log(settings.MIN_RULE_FREQ)
	return compute_logfreq

def ruleprob(rule_prob_file):
	rule_probs = {}
	for r, freq, c1, c2, p in load_tsv_file(rule_prob_file):
		rule_probs[r] = float(p)
	def prob(rule):
		return rule_probs[rule] if rule_probs.has_key(rule) else 0.0
	return prob

def dict_weight(rules):
	def weight(rule):
		if rules.has_key(rule):
			return rules[rule]
		return 0.0
	return weight

def load_graph_with_rules_cl(graph_file, rule_cl, weight_fun):
	graph = Graph()
	for row in load_tsv_file(graph_file):
		rule = row[2]
		v1 = (row[0], rule_cl[rule][0])
		v2 = (row[1], rule_cl[Rule.from_string(rule).reverse().to_string()][0])
		w = weight_fun(rule)
		graph.add_vertex(v1)
		graph.add_vertex(v2)
		graph.add_edge(v1, v2, w)
	return graph

# TODO write results to the graph line by line
# -> separate function: cluster_edges
def load_graph_with_local_edges_cl(graph_file, s_rul_cooc_file, weight_fun):
	# load surface rules MI
	s_rul_mi = {}
	for r1, r2, freq, mi in load_tsv_file(s_rul_cooc_file):
		if mi > settings.INDEPENDENCY_THRESHOLD:
			s_rul_mi[(r1, r2)] = mi
	# load graph and cluster edges on fly
	word_edgeset, clusterings = {}, {}

	def cluster_edges_set(word, edges):
		edges_str = ','.join(sorted([r for w, r in edges]))
		word_edgeset[word] = edges_str
		if not clusterings.has_key(edges_str):
			# build graph from edges
			edges_gr = Graph(edges_storage=datastruct.storage.DICT)
			for w1, r1 in edges:
				edges_gr.add_vertex(r1)
				for w2, r2 in edges:
					edges_gr.add_vertex(r2)
					if r1 < r2:
						if s_rul_mi.has_key((r1, r2)):
							edges_gr.add_edge(r1, r2, s_rul_mi[(r1, r2)])
							edges_gr.add_edge(r2, r1, s_rul_mi[(r1, r2)])
			# cluster the edges graph
			edges_cl = algorithms.clustering.clink(edges_gr)
			edges_cl = algorithms.clustering.clusters_list_to_dict(edges_cl)
			# memorize results
			clusterings[edges_str] = edges_cl
#			print ' ; '.join(edges_cl)

	cur_word, edges = None, []
	pp = progress_printer(get_file_size(graph_file))
	for word, edges in load_tsv_file_by_key(graph_file, 1):
		cluster_edges_set(word, edges)
		for i in range(0, len(edges)):
			pp.next()
	if edges:
		cluster_edges_set(cur_word, edges)
	# load graph and use computed clusterings
	graph_p = Graph()
	for word_1, word_2, rule in load_tsv_file(graph_file):
		v1 = (word_1, clusterings[word_edgeset[word_1]][rule][0])
		v2 = (word_2, clusterings[word_edgeset[word_2]][\
			Rule.from_string(rule).reverse().to_string()][0])
		w = weight_fun(rule)
		graph_p.add_vertex(v1)
		graph_p.add_vertex(v2)
		graph_p.add_edge(v1, v2, w)
	graph_p.save_to_file('graph_p.txt')
	return graph_p

### LEXEMES POSTPROCESSING ###

# TODO czy w razie duplikatu nie jest tak, ze rozne slowa wybieraja rozne kopie leksemu?
#  moze i nie, jesli max() jest stabilna? (sprawdzic!)
def best_cluster_for_word(graph, lexemes):
	# for every word, determine, which cluster is the best for it
	cl_scores = {}
	for lex in lexemes:
		for word, cl_id in lex:
			score = 0
			for word_2, cl_id_2 in lex:
				if graph.has_edge((word, cl_id), (word_2, cl_id_2)):
					score += graph.get_edge_weight((word, cl_id), (word_2, cl_id_2))
			if not cl_scores.has_key(word):
				cl_scores[word] = []
			cl_scores[word].append((cl_id, score))
	for word in cl_scores.keys():
		cl_scores[word] = max(cl_scores[word], key = lambda x: x[1])[0]
	# remove this word from all other clusters
	new_lexemes = []
	for lex in lexemes:
		new_lex = []
		for word, cl_id in lex:
			if cl_id == cl_scores[word]:
				new_lex.append(word)
		if new_lex:
			new_lexemes.append(new_lex)
	return new_lexemes

def remove_subsets(lexemes):
	# TODO remove only lexemes, that are subsets of others
	pass

### PARADIGMS ###

def extract_paradigm(lexeme, word):
	paradigm = []
	for word_2 in lexeme:
		if word_2 != word:
			rule = algorithms.align.align(word, word_2)
			paradigm.append(rule.to_string())
	return paradigm

def extract_paradigms_from_lexemes(lexemes):
	word_par = {}
	i_rules = Counter()
	pp = progress_printer(len(lexemes))
	for lex in lexemes:
		for word in lex:
			paradigm = extract_paradigm(lex, word)
			for i_rule in paradigm:
				i_rules.inc(i_rule)
			word_par[word] = ','.join(sorted(paradigm)) if paradigm else '-'
		pp.next()
	return word_par, i_rules

### LEMMATIZATION ###

def lemmatize(lexemes, word_freq, word_par):
	word_lemma = {}
	# score paradigms (sum up freq. of words having this paradigm)
	paradigms_score = Counter()
	for word, paradigm in word_par.iteritems():
		paradigms_score.inc(paradigm, word_freq[word])
	# within each lexeme, choose the word which has the highest-scored paradigm
	for lex in lexemes:
		lemma = max(lex, key = lambda w: paradigms_score[word_par[w]])
		for word in lex:
			word_lemma[word] = lemma
	return word_lemma

### SAVING RESULTS ###

def save_inflection(lexemes, word_lemma, word_par, i_rules):
	# add words that do not occur in the graph
	for word, freq in load_tsv_file(settings.FILES['wordlist']):
		if not word_lemma.has_key(word):
			word_lemma[word] = word
		if not word_par.has_key(word):
			word_par[word] = '-'
	# - lexemes
	with open_to_write(settings.FILES['inflection.lexemes']) as fp:
		for lex in lexemes:
			write_line(fp, (', '.join(sorted(lex)), ))
	sort_file(settings.FILES['inflection.lexemes'])
	# - inflection
	with open_to_write(settings.FILES['inflection']) as fp:
		for word in word_lemma.keys():
			write_line(fp, (word, word_lemma[word], word_par[word]))
	sort_file(settings.FILES['inflection'])
	# - paradigms / freq
	paradigms_c = Counter()
	for paradigm in word_par.values():
		paradigms_c.inc(paradigm)
	paradigms_c.save_to_file(settings.FILES['inflection.paradigms'])
	# - infl. rules
	i_rules.save_to_file(settings.FILES['inflection.rules'])

### TRAINING DATA ###

def load_trained_infl_rules(filename):
	i_rules_c = {}
	print 'Loading inflectional rules...'
	for rule, ifreq, freq, weight in load_tsv_file(filename):
		i_rules_c[rule] = float(weight)
	return i_rules_c

## MAIN FUNCTIONS ###

def run():
#	print 'Clustering surface rules...'
#	rule_cl_lst = cluster_rules(settings.SURFACE_RULES_FILE,\
#		settings.SURFACE_RULES_COOC_FILE)
#	rule_cl = algorithms.clustering.clusters_list_to_dict(rule_cl_lst)

	if file_exists(settings.FILES['surface.graph.partitioned']):
		print 'Loading partitioned graph...'
		graph = Graph.load_from_file(settings.FILES['surface.graph.partitioned'],\
			(0, 1), (2, 3), 4)
	elif file_exists(settings.FILES['trained.rules']):	# supervised
		print 'Loading graph for supervised clutering...'
		i_rules = load_trained_infl_rules(settings.FILES['trained.rules'])
		graph = load_graph_with_local_edges_cl(settings.FILES['surface.graph'],
			settings.FILES['trained.rules.cooc'],
			dict_weight(i_rules))
	else:	# unsupervised
		print 'Loading graph for unsupervised clutering...'
#		graph = load_graph_with_local_edges_cl(settings.FILES['surface.graph'],
#			settings.FILES['surface.rules.cooc'],
#			ruleprob('s_rul_prob.txt'))
		graph = load_graph_with_local_edges_cl(settings.FILES['surface.graph'],
			settings.FILES['surface.rules.cooc'],
			logfreq(Counter.load_from_file(settings.FILES['surface.rules'])))
#		graph = load_graph_with_local_edges_cl(settings.FILES['surface.graph'],
#			settings.FILES['surface.rules.cooc'],
#			entropy(Counter.load_from_file(settings.FILES['surface.rules']),\
#				get_file_size(settings.FILES['wordlist'])))
	print 'Clustering graph...'
	lexemes = algorithms.clustering.chinwhisp(graph, 0.0, 20)	# TODO threshold

#	ruleprob_fun = ruleprob('s_rul_prob.txt')
#	logfreq_fun = logfreq(Counter.load_from_file(settings.FILES['surface.rules']))
#	graph = Graph.load_from_file(settings.FILES['surface.graph'], 0, 1,
#		lambda row: logfreq_fun(row[2]))
#	print 'Clustering graph...'
#	lexemes = algorithms.clustering.chinwhisp_prob(graph, 0.0, 20)
#	lexemes = algorithms.clustering.clink(graph)

	print 'Post-processing lexemes...'
	lexemes = best_cluster_for_word(graph, lexemes)
#	with open_to_write(settings.FILES['inflection.lexemes']) as fp:
#		for lex in lexemes:
#			write_line(fp, (', '.join(sorted(lex)), ))
###			write_line(fp, (', '.join(sorted([l[0]+'('+str(l[1])+')' for l in lex])), ))
#	sort_file(settings.FILES['inflection.lexemes'])
	print 'Extracting paradigms...'
	word_par, i_rules = extract_paradigms_from_lexemes(lexemes)
	print 'Lemmatizing...'
	word_freq = Counter.load_from_file(settings.FILES['wordlist'])
	word_lemma = lemmatize(lexemes, word_freq, word_par)
	print 'Saving results...'
	save_inflection(lexemes, word_lemma, word_par, i_rules)

def evaluate():
	words = []
	for word, freq in load_tsv_file(settings.FILES['wordlist']):
		words.append(word)

	def load_clusters(filename, key_col, value_col):
		# TODO don't evaluate words with empty paradigm?
		clusters = {}
		for row in load_tsv_file(filename):
			if not clusters.has_key(row[key_col-1]):
				clusters[row[key_col-1]] = []
			clusters[row[key_col-1]].append(row[value_col-1])
		return clusters
	
	def load_lemmas(filename):
		# multiple lemmas for a word -> count as correct if all lemmas found
		lemmas = {}
		for row in load_tsv_file(filename):
			word, lemma = row[0], row[1]
			if not lemmas.has_key(word):
				lemmas[word] = []
			lemmas[word].append(lemma)
		return lemmas

	# evaluate lexemes clustering
	clusters = algorithms.clustering.clusters_dict_to_list(\
		load_clusters(settings.FILES['inflection'], 1, 2))
	classes = algorithms.clustering.clusters_dict_to_list(\
		load_clusters(settings.FILES['inflection.eval'], 1, 2))
	pre, rec, f_sc = algorithms.clustering.bcubed(words, clusters, classes,\
		settings.FILES['inflection.eval.log'])
	print 'LEXEMES CLUSTERING:'
	print 'Precision: %0.2f %%' % (100*pre)
	print 'Recall: %0.2f %%' % (100*rec)
	print 'F-score: %0.2f %%\n' % (100*f_sc)
	# evaluate paradigms clustering?
	# evaluate lemmatization
#	lemmas = load_lemmas(settings.FILES['inflection'])
#	gs_lemmas = load_lemmas(settings.FILES['inflection.eval'])
#	correct, total = 0, 0
#	for word in words:
#		if gs_lemmas.has_key(word):
#			total += 1
#			if set(lemmas[word]) == set(gs_lemmas[word]):
#				correct += 1
#	print '\nLEMMAS EVALUATION'
#	print 'Correct: %0.2f %%\n' % (100*float(correct) / total)

def import_from_db():
	utils.db.connect()
	print 'Importing inflectional rules...'
	utils.db.pull_table(settings.I_RUL_TABLE, ('rule', 'freq'),\
		settings.FILES['inflection.rules'])
	print 'Importing paradigms...'
	# paradigms: with nested query, make a dict id => par_str
	paradigms = {}
	with open_to_write(settings.FILES['inflection.paradigms']) as fp:
		for p_id, freq in utils.db.query_fetch_all_results(\
			'SELECT p_id, freq FROM paradigms;'):
			par_rules = []
			for (rule, ) in utils.db.query_fetch_all_results('''
				SELECT r.rule FROM par_rul p 
					JOIN i_rul r ON p.r_id = r.r_id
					WHERE p.p_id = %d
				;''' % p_id):
				par_rules.append(rule)
			par_str = ','.join(sorted(par_rules)) if par_rules else '-'
			paradigms[p_id] = par_str
			write_line(fp, (par_str, freq))
	# inflection: substitute lemma and paradigm IDs with values
	print 'Importing inflectional analyses...'
	with open_to_write(settings.FILES['inflection']) as fp:
		for word, lemma, p_id in utils.db.query_fetch_results('''
			SELECT w.word, lem.word, i.p_id FROM inflection i
				JOIN words w ON i.w_id = w.w_id
				JOIN lexemes l ON i.l_id = l.l_id
				JOIN words lem ON l.lemma = lem.w_id
			;'''):
			write_line(fp, (word, lemma, paradigms[p_id]))
	utils.db.close_connection()

def export_to_db():
	# get word IDs from the database (needed for converting inflection)
	utils.db.connect()
	word_ids = {}
	for word, w_id in utils.db.query_fetch_results('SELECT word, w_id FROM words;'):
		word_ids[word] = w_id
	# infl. rules <- insert ID
	print 'Converting inflectional rules...'
	i_rule_ids = utils.db.insert_id(settings.FILES['inflection.rules'],\
		settings.FILES['inflection.rules.db'])
	# paradigms <- insert ID, move infl. rules to a separate file, replace infl. rules with their IDs
	print 'Converting paradigms...'
	par_ids = {}
	pp = progress_printer(get_file_size(settings.FILES['inflection.paradigms']))
	with open_to_write(settings.FILES['inflection.paradigms.db']) as par_fp:
		with open_to_write(settings.FILES['inflection.par_rul.db']) as par_mem_fp:
			for i, (paradigm, freq) in enumerate(\
				load_tsv_file(settings.FILES['inflection.paradigms']), 1):
				par_ids[paradigm] = i
				write_line(par_fp, (i, freq, len(paradigm.split(',')), paradigm))
				pp.next()
				if paradigm == '-': continue		# empty paradigm
				for i_rule in paradigm.split(','):
					write_line(par_mem_fp, (i, i_rule_ids[i_rule]))
	# give IDs to lexemes
	print 'Converting lexemes...'
	lex_ids = {}
	with open_to_write(settings.FILES['inflection.lexemes.db']) as fp:
		for i, (lemma, words) in enumerate(\
			load_tsv_file_by_key(settings.FILES['inflection'], 2), 1):
			lex_ids[lemma] = i
			write_line(fp, (i, word_ids[lemma], len(words)))
	# inflection
	print 'Converting inflectional analyses...'
	utils.db.replace_values_with_ids(settings.FILES['inflection'],\
		settings.FILES['inflection.db'], (word_ids, lex_ids, par_ids))
	# load tables into DB
	print 'Exporting inflectional rules...'
	utils.db.push_table(settings.I_RUL_TABLE, settings.FILES['inflection.rules.db'])
	print 'Exporting lexemes...'
	utils.db.push_table(settings.LEXEMES_TABLE, settings.FILES['inflection.lexemes.db'])
	print 'Exporting paradigms...'
	utils.db.push_table(settings.PARADIGMS_TABLE, settings.FILES['inflection.paradigms.db'])
	utils.db.push_table(settings.PAR_RUL_TABLE, settings.FILES['inflection.par_rul.db'])
	print 'Exporting inflectional analyses...'
	utils.db.push_table(settings.INFLECTION_TABLE, settings.FILES['inflection.db'])
	utils.db.close_connection()
	# delete temporary files
	remove_file(settings.FILES['inflection.rules.db'])
	remove_file(settings.FILES['inflection.lexemes.db'])
	remove_file(settings.FILES['inflection.paradigms.db'])
	remove_file(settings.FILES['inflection.par_rul.db'])
	remove_file(settings.FILES['inflection.db'])

