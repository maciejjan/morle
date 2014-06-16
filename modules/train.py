import algorithms.align
import algorithms.cooccurrences
import algorithms.fastss
from datastruct.counter import *
from utils.files import *
from utils.printer import *
import settings
import re

def load_lexemes(filename):
	lexemes = []
	for lemma, words in load_tsv_file_by_key(filename, 2):
		lexemes.append([w[0] for w in words])
	return lexemes

def extract_rules_from_lexemes(lexemes, output_file):
	print 'Extracting rules from lexemes...'
	pp = progress_printer(len(lexemes))
	with open_to_write(output_file) as outfp:
		for lexeme in lexemes:
			for w1 in lexeme:
				for w2 in lexeme:
					if w1 < w2:
						rule = algorithms.align.align(w1, w2)
						if rule is not None:
							write_line(outfp, (w1, w2, rule.to_string()))
							write_line(outfp, (w2, w1, rule.reverse().to_string()))
			pp.next()

def save_lexemes_as_wordlist(lexemes, output_file):
	with open_to_write(output_file) as outfp:
		for lex in lexemes:
			for word in lex:
				write_line(outfp, (word, 1))

def extract_rules_from_substrings(substrings_file, graph_file, i_rules):
	print 'Extracting rules from substrings...'
	pp = progress_printer(get_file_size(substrings_file))
	with open_to_write(graph_file) as outfp:
		for substr, rows in load_tsv_file_by_key(substrings_file, 2):
			pattern = re.compile('(.*)' +\
				'(.*?)'.join([letter for letter in substr]) + '(.*)')
			for (s_len_1, w1, w_freq_1) in rows:
				for (s_len_2, w2, w_freq_2) in rows:
					if w1 < w2:
						rule = algorithms.align.extract_rule(w1, w2, pattern)
						if rule is not None and i_rules.has_key(rule.to_string()):
							write_line(outfp, (w1, w2, rule.to_string()))
							write_line(outfp, (w2, w1, rule.reverse().to_string()))
			for i in range(len(rows)):
				pp.next()

def count_rules(graph_file):
	rules_c = Counter()
	for rule, wordpairs in load_tsv_file_by_key(graph_file, 3):
		rules_c.add(rule, len(wordpairs))
	return rules_c

def build_surface_graph(wordlist_file, substrings_file, graph_file, i_rules):
	algorithms.fastss.create_substrings_file(wordlist_file, substrings_file)
	extract_rules_from_substrings(substrings_file, graph_file, i_rules)

def save_rule_weights(i_rules_c, s_rules_c, output_file):
	with open_to_write(output_file) as outfp:
		for rule in i_rules_c.keys():
			if s_rules_c.has_key(rule) and s_rules_c[rule] > i_rules_c[rule]:
				write_line(outfp, (rule, i_rules_c[rule], s_rules_c[rule], \
					float(i_rules_c[rule]) / s_rules_c[rule]))
			else:
				write_line(outfp, (rule, i_rules_c[rule], 
					0 if not s_rules_c.has_key(rule) else s_rules_c[rule], \
					1.0))		# TODO ???

def run():
	sort_file(settings.FILES['training.inflection'], key=2)
	lexemes = load_lexemes(settings.FILES['training.inflection'])
	extract_rules_from_lexemes(lexemes, settings.FILES['training.inflection.graph'])
	sort_file(settings.FILES['training.inflection.graph'], key=3)
	i_rules_c = count_rules(settings.FILES['training.inflection.graph'])
	save_lexemes_as_wordlist(lexemes, settings.FILES['training.wordlist'])
	build_surface_graph(settings.FILES['training.wordlist'],\
		settings.FILES['training.substrings'],\
		settings.FILES['training.surface.graph'], i_rules_c)
	sort_file(settings.FILES['training.surface.graph'], key=3)
	s_rules_c = count_rules(settings.FILES['training.surface.graph'])
	save_rule_weights(i_rules_c, s_rules_c, \
		settings.FILES['trained.rules'])
	sort_file(settings.FILES['training.inflection.graph'], key=1)
	algorithms.cooccurrences.calculate_rules_cooc(\
		settings.FILES['training.inflection.graph'],\
		settings.FILES['trained.rules.cooc'], i_rules_c)

def evaluate():
	pass

def import_from_db():
	utils.db.connect()
	print 'Importing surface rules...'
	utils.db.pull_table(settings.TR_RUL_TABLE, ('rule', 'ifreq', 'freq', 'weight'),\
		settings.FILES['trained.rules'])
	# pull surface rules co-occurrences
	print 'Importing trained rules co-occurrences...'
	with open_to_write(settings.FILES['trained.rules.cooc']) as fp:
		for rule_1, rule_2, freq, sig in utils.db.query_fetch_results('''
			SELECT r1.rule, r2.rule, c.freq, c.sig FROM tr_rul_co c
				JOIN tr_rul r1 ON c.r1_id = r1.r_id
				JOIN tr_rul r2 ON c.r2_id = r2.r_id
			;'''):
			write_line(fp, (rule_1, rule_2, freq, sig))
	utils.db.close_connection()

def export_to_db():
	print 'Converting surface rules...'
	rule_ids = utils.db.insert_id(settings.FILES['trained.rules'],\
		settings.FILES['trained.rules.db'])
	print 'Converting surface rules co-occurrences...'
	utils.db.replace_values_with_ids(settings.FILES['trained.rules.cooc'],\
		settings.FILES['trained.rules.cooc.db'],\
		(rule_ids, rule_ids, None, None))
	# load tables into DB
	utils.db.connect()
	print 'Exporting trained rules...'
	utils.db.push_table(settings.TR_RUL_TABLE, settings.FILES['trained.rules.db'])
	print 'Exporting trained rule co-occurrences...'
	utils.db.push_table(settings.TR_RUL_CO_TABLE,\
		settings.FILES['trained.rules.cooc.db'])
	utils.db.close_connection()
	# delete temporary files
	remove_file(settings.FILES['trained.rules.db'])
	remove_file(settings.FILES['trained.rules.cooc.db'])

