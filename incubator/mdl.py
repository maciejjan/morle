from datastruct.counter import *
import incubator.ngrams as ngr
from utils.files import *

N = 1
CORPUS_SIZE = 267210
NUM_ITERATIONS = 1

def load_rules_p(rules_p_file):
	rules_p = {}
	for rule, f, f_l in load_tsv_file(rules_p_file,\
			print_progress=True, print_msg='Loading rule productivity...'):
		rules_p[rule] = float(f) / float(f_l)
	return rules_p

def annotate_graph(input_file, output_file, ngram_model, rules_p):
	with open_to_write(output_file) as outfp:
		for word_1, word_2, rule in load_tsv_file(input_file,\
				print_progress=True, print_msg='Annotating graph...'):
			if not rules_p.has_key(rule):
				continue
			n_gr = int(rules_p[rule] /\
				(ngr.word_prob(word_2, ngram_model, N) * (1-rules_p[rule])))
			write_line(outfp, (word_1, word_2, rule, rules_p[rule], ngr.word_prob(word_2, ngram_model, N), n_gr))

def not_derived(w_1, w_2, derived):
	if not derived.has_key(w_1):
		return True
	elif derived[w_1] == w_2:
		return False
	else:
		return not_derived(derived[w_1], w_2, derived)

def filter_edges(input_file, output_file):
	derived = {}
	n = CORPUS_SIZE
	with open_to_write(output_file) as outfp:
		for word_1, word_2, rule, rp, wp, n_gr in load_tsv_file(input_file,\
				print_progress=True, print_msg='Filtering graph edges...'):
			n_gr = int(n_gr)
			if n <= n_gr and not derived.has_key(word_2) and\
					 not_derived(word_1, word_2, derived):
				if word_2 == u'mah':
					print word_1, n, n_gr, n <= n_gr
				derived[word_2] = word_1
				write_line(outfp, (word_1, word_2, rule, rp, wp, n_gr))
				n -= 1
	print 'n =', n

#def reestimate_prod(input_file, output_file):
#	rules_c = Counter()
#	with open_to_write(output_file) as outfp:
#		for word_1, word_2, rule, rp, wp, n_gr in load_tsv_file(input_file,\
#				print_progress=True, print_msg='Re-estimating rule productivity...'):
#			rules_c.inc(rule)
#	rules_c.save_to_file(output_file)

def reestimate_prod(input_file, rules_p_file, output_file):
	rules_c = Counter()
	with open_to_write(output_file) as outfp:
		for word_1, word_2, rule, rp, wp, n_gr in load_tsv_file(input_file,\
				print_progress=True, print_msg='Re-estimating rule productivity...'):
			rules_c.inc(rule)
		for rule, f, f_l in load_tsv_file(rules_p_file):
			if rules_c.has_key(rule):
				write_line(outfp, (rule, rules_c[rule], f_l))

def load_graph(input_file):
	derived = {}
	for word_1, word_2, rule, rp,wp, n_gr in load_tsv_file(input_file,\
			print_progress=True, print_msg='Loading graph...'):
		derived[word_2] = word_1
	return derived

def get_path(word, derived):
	path = [word]
	while derived.has_key(word):
		word = derived[word]
		path.append(word)
	return path

def analyze_all(input_file, output_file, derived):
	with open_to_write(output_file) as outfp:
		for word, freq in load_tsv_file(input_file,\
				print_progress=True, print_msg='Analyzing words...'):
			path = get_path(word, derived)
			write_line(outfp, (word, '<- ' + ' <- '.join(path[1:])))

def run(ngram_model, iter_num):
	# load rules and rule productivities
	rules_p = load_rules_p('rule_prod.txt')
	# annotate edges with bordering N value
	annotate_graph('graph.txt', 'graph_mdl.txt', ngram_model, rules_p) # sort edges
	sort_file('graph_mdl.txt', key=6, reverse=True, numeric=True)
	# filter edges
	filter_edges('graph_mdl.txt', 'graph_mdl_fil.txt')
	# reestimate rule probability
	rename_file('rule_prod.txt', 'rule_prod.txt.' + str(iter_num))
	reestimate_prod('graph_mdl_fil.txt', 'rule_prod.txt.' + str(iter_num), 'rule_prod.txt')
	# analyze input
	d = load_graph('graph_mdl_fil.txt')
	analyze_all('input.txt', 'analyzed.txt.'+str(iter_num), d)

def em():
	# train n-gram model
	ngram_model = ngr.train('input.txt', N)
	ngram_model.normalize()
	# run the EM estimation
	for i in range(NUM_ITERATIONS):
		print '\nIteration', i+1
		run(ngram_model, i+1)

