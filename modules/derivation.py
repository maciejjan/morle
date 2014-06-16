#from datastruct.counter import *
#from datastruct.rules import *
import utils.db
from utils.files import *
from utils.printer import *
import settings

def ruleprob(S_RUL_PROB_FILE):
	ruleprob_dict = {}
	for row in load_tsv_file(S_RUL_PROB_FILE):
#		ruleprob_dict[row[0]] = float(row[1]) / float(row[2])
		ruleprob_dict[row[0]] = float(row[2])

	def ruleprob_fun(rule):
		return ruleprob_dict[rule] if ruleprob_dict.has_key(rule) else 0.0
	return ruleprob_fun

def load_word_lex(INFLECTION_FILE):
	word_lex = {}
	for word, lex, par in load_tsv_file(INFLECTION_FILE):
		word_lex[word] = lex
	return word_lex

def run():
	word_lex = load_word_lex(settings.FILES['inflection'])
	ruleprob_fun = ruleprob(settings.FILES['surface.rules'])
	word_max, lex_max = {}, {}
	pp = progress_printer(get_file_size(settings.FILES['surface.graph']))
	for word_1, word_2, rule in load_tsv_file(settings.FILES['surface.graph']):
		pp.next()
		if word_lex[word_1] == word_lex[word_2]:
			continue
		if not word_max.has_key(word_1) or ruleprob_fun(rule) > word_max[word_1][1]:
			word_max[word_1] = (word_2, ruleprob_fun(rule), rule)
		if word_lex.has_key(word_1) and word_lex.has_key(word_2):
			lex_1, lex_2 = word_lex[word_1], word_lex[word_2]
			if not lex_max.has_key(lex_1) or ruleprob_fun(rule) > lex_max[lex_1][1]:
				lex_max[lex_1] = (lex_2, ruleprob_fun(rule), rule)
	with open_to_write(settings.FILES['derivation']) as fp:
		for lex in lex_max.keys():
			if lex_max[lex][1] > settings.DERIVATION_THRESHOLD:
				write_line(fp, (lex, lex_max[lex][0], lex_max[lex][1], lex_max[lex][2]))

def evaluate():
	log_file = codecs.open(settings.FILES['derivation.eval.log'], 'w+', settings.ENCODING)

	# load sets of words for each lemma
	lemma_words = {}
	for row in load_tsv_file(settings.FILES['inflection']):
		word, lemma = row[0], row[1]
		if not lemma_words.has_key(lemma):
			lemma_words[lemma] = []
		lemma_words[lemma].append(word)
	# load golden standard lemmatization
	gs_lemmas = {}
	for row in load_tsv_file(settings.FILES['inflection.eval']):
		word, lemma = row[0], row[1]
		if not gs_lemmas.has_key(word):
			gs_lemmas[word] = []
		gs_lemmas[word].append(lemma)
	# load golden standard derivations
	derivations = set([])
	for lemma_1, lemma_2 in load_tsv_file(settings.FILES['derivation.eval']):
		derivations.add((lemma_1, lemma_2))
	# load derivations and evaluate
	found, count = set([]), 0
	for row in load_tsv_file(settings.FILES['derivation']):
		lemma_1, lemma_2 = row[0], row[1]
		if not lemma_words.has_key(lemma_1) or not lemma_words.has_key(lemma_2):
			continue
		real_lemmas_1, real_lemmas_2 = [], []
		for w in lemma_words[lemma_1]:
			if gs_lemmas.has_key(w):
				real_lemmas_1.extend(gs_lemmas[w])
		for w in lemma_words[lemma_2]:
			if gs_lemmas.has_key(w):
				real_lemmas_2.extend(gs_lemmas[w])
		if not real_lemmas_1 or not real_lemmas_2:
			continue
		found_pair, already_found = None, False
		for rl_1 in real_lemmas_1:
			for rl_2 in real_lemmas_2:
				if (rl_1, rl_2) in found or (rl_2, rl_1) in found:
					already_found = True
				if (rl_1, rl_2) in derivations and not (rl_1, rl_2) in found:
					found_pair = (rl_1, rl_2)
					break
				if (rl_2, rl_1) in derivations and not (rl_2, rl_1) in found:
					found_pair = (rl_2, rl_1)
					break
		if found_pair is not None:
			found.add(found_pair)
			log_file.write("TP: %s, %s (%s, %s)\n" %\
				(lemma_1, lemma_2, found_pair[0], found_pair[1]))
			count += 1
		elif not already_found:
			count += 1
			if set(real_lemmas_1) & set(real_lemmas_2):
				log_file.write("FP (Infl): %s, %s\n" % (lemma_1, lemma_2))
			else:
				log_file.write("FP: %s, %s\n" % (lemma_1, lemma_2))
	for (lemma_1, lemma_2) in derivations:
		if not (lemma_1, lemma_2) in found:
			log_file.write("FN: %s, %s\n" % (lemma_1, lemma_2))
	# print results
	pre = float(len(found)) / count
	rec = float(len(found)) / len(derivations)
	fsc = 2*pre*rec / (pre + rec)
	print '\n\nDERIVATION EVALUATION:'
	print 'Precision: %0.2f %% ' % (100 * pre)
	print 'Recall: %0.2f %% ' % (100 * rec)
	print 'F-score: %0.2f %% \n' % (100 * fsc)
	log_file.close()

def import_from_db():
	utils.db.connect()
	# create a dictionary id => str_val for paradigms
	paradigms = {}
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
	# import derivational rules
	print 'Importing derivational rules...'
	with open_to_write(settings.FILES['derivation.rules']) as fp:
		for rule, p1_id, p2_id, freq in utils.db.query_fetch_results('''
			SELECT r.rule, d.p1_id, d.p2_id, d.freq FROM derivation d
				JOIN s_rul r ON d.r_id = r.r_id
			;'''):
			write_line(fp, (rule, paradigms[p1_id], paradigms[p2_id], freq))
	# import derivation
	print 'Importing derivation...'
	with open_to_write(settings.FILES['derivation']) as fp:
		for word_1, word_2 in utils.db.query_fetch_results('''
			SELECT w1.word, w2.word FROM derivation d
				JOIN words w1 ON d.w1_id = w1.w_id
				JOIN words w2 ON d.w2_id = w2.w_id
			;'''):
			write_line(fp, (word_1, word_2))
	utils.db.close_connection()

def export_to_db():
	utils.db.connect()
	# get IDs of lexemes
	lex_ids = {}
	for lemma, l_id in utils.db.query_fetch_results('''
			SELECT lem.word, l.l_id FROM lexemes l
				JOIN words lem ON l.lemma = lem.w_id;'''):
		lex_ids[lemma] = l_id
	# get the IDs of surface rules, inflectional rules and paradigms
	s_rul_ids = {}
	for rule, r_id in utils.db.query_fetch_results('SELECT rule, r_id FROM s_rul;'):
		s_rul_ids[rule] = r_id
	print 'Converting derivation...'
	with open_to_write(settings.FILES['derivation.db']) as fp:
		for l1, l2, prob, rule in \
			load_tsv_file(settings.FILES['derivation']):
			write_line(fp, (lex_ids[l1], lex_ids[l2], s_rul_ids[rule]))
	# load table into DB
	print 'Exporting derivation...'
	utils.db.push_table(settings.DERIVATION_TABLE, settings.FILES['derivation.db'])
	utils.db.close_connection()
	# delete temporary file
	remove_file(settings.FILES['derivation.db'])

