# -*- encoding: utf-8 -*-

import algorithms.align
import algorithms.fastss
import algorithms.cooccurrences
from algorithms.ngrams import NGramModel
import algorithms.optrules
from datastruct.counter import *
from datastruct.rules import *
from datastruct.lexicon import *
import utils.db
from utils.files import *
from utils.printer import *
import settings
import re
import random

### LOCAL FILTERS ###

# form: filter(rule, (wordpair))
def _lfil_affix_length(rule, wordpair):
	affixes = rule.get_affixes()
	if not affixes:
		return True
	if len(affixes) == 1 and not rule.alternations: # exception for single-affix rules
		return True
	if max([len(a) for a in affixes]) <= settings.MAX_AFFIX_LENGTH:
		return True
	return False

LOCAL_FILTERS = [_lfil_affix_length]

def apply_local_filters(rule, wordpair):
	for f in LOCAL_FILTERS:
		if not f(rule, wordpair):
			return False
	return True

### GLOBAL FILTERS ###

# form: filter(rule, [list, of, wordpairs...]) 
def _gfil_rule_freq(rule, wordpairs):
	return len(wordpairs) >= settings.MIN_RULE_FREQ

GLOBAL_FILTERS = [_gfil_rule_freq]

def apply_global_filters(rule, wordpairs):
	for f in GLOBAL_FILTERS:
		if not f(rule, wordpairs):
			return False
	return True

### TRAINING DATA ###

def load_training_infl_rules(filename):
	i_rules_c = {}
	print('Loading inflectional rules...')
	for rule, ifreq, freq, weight in read_tsv_file(filename):
		i_rules_c[rule] = weight
	return i_rules_c

### RULE EXTRACTING FUNCTIONS ###

def extract_rules_from_words(words, substring, outfp, wordset):
	pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in substring]) + '(.*)')
	for w1, w1_freq in words:
		for w2, w2_freq in words:
#			if w1 < w2:
			rule = algorithms.align.extract_rule(w1, w2, pattern)
			if rule is not None:
				if apply_local_filters(rule, ((w1, w1_freq), (w2, w2_freq))):
					write_line(outfp, (w1, w2, rule.to_string()))
#					write_line(outfp, (w2, w1, rule.reverse().to_string()))
				elif settings.COMPOUNDING_RULES:
					for i in range(3, min(len(w1), len(w2))):
						if i <= len(rule.prefix[1]) and rule.prefix[1][:i] in wordset:
							write_line(outfp, (w1, w2, rule.to_string()))
							break
						if i <= len(rule.suffix[1]) and rule.suffix[1][-i:] in wordset:
							write_line(outfp, (w1, w2, rule.to_string()))
							break

def extract_rules_from_substrings(input_file, output_file, wordset=None):
	cur_substr, words = '', []
	pp = progress_printer(get_file_size(input_file))
	with open_to_write(output_file) as outfp:
		for substr, rest in read_tsv_file_by_key(input_file, 2, print_progress=True,\
				print_msg='Extracting rules from substrings...'):
			words = [(word, freq) for (s_len, word, freq) in rest]
			extract_rules_from_words(words, substr, outfp, wordset)

def filter_lemmas(graph_file):
	lemmas = set()
	for word, freq, base in read_tsv_file('lexicon.full', (str, int, str),\
			print_progress=True, print_msg='Loading lemmas...'):
		lemmas.add(base)
	rename_file('graph.txt', 'graph.txt.unfil')
	update_file_size('graph.txt.unfil')
	with open_to_write('graph.txt') as outfp:
		for word_1, word_2, rule in read_tsv_file('graph.txt.unfil', (str, str, str),\
				print_progress=True, print_msg='Filtering...'):
			if word_1 in lemmas and word_2 not in lemmas:	# TODO analiza -> analizy not in graph!
				write_line(outfp, (word_1, word_2, rule))
	update_file_size('graph.txt')

def filter_rules(graph_file):
	'''Filter rules according to frequency.'''
	# Annotate graph with rule frequency
	# format: w1 w2 rule -> w1 w2 rule freq
	with open_to_write(graph_file + '.tmp') as graph_tmp_fp:
		for rule, wordpairs in read_tsv_file_by_key(graph_file, 3, 
				print_progress=True, print_msg='Counting rule frequency...'):
			freq = len(wordpairs)
			for w1, w2 in wordpairs:
				write_line(graph_tmp_fp, (w1, w2, rule, freq))
	# sort rules according to frequency
	sort_file(graph_file + '.tmp', stable=True, numeric=True, reverse=True, key=4)
	# truncate the graph file to most frequent rules
	print('Filtering rules...')
	pp = progress_printer(settings.MAX_NUM_RULES)
	with open_to_write(graph_file + '.filtered') as graph_fil_fp:
		num_rules = 0
		for (rule, freq), wordpairs in read_tsv_file_by_key(graph_file + '.tmp', (3, 4)):
			try:
				next(pp)
			except StopIteration:
				break
			for w1, w2 in wordpairs:
				write_line(graph_fil_fp, (w1, w2, rule))
			num_rules += 1
	# cleanup files
	remove_file(graph_file + '.tmp')
	rename_file(graph_file, graph_file + '.orig')
	rename_file(graph_file + '.filtered', graph_file)

def load_wordset(input_file):
	wordset = set([])
	for word, freq in read_tsv_file(input_file):
		wordset.add(word)
	return wordset

### MAIN FUNCTIONS ###

def run():
	wordset = load_wordset(settings.FILES['training.wordlist'])\
		if settings.COMPOUNDING_RULES else None
	algorithms.fastss.create_substrings_file(\
		settings.FILES['training.wordlist'], settings.FILES['surface.substrings'], wordset)
	extract_rules_from_substrings(settings.FILES['surface.substrings'],\
		settings.FILES['surface.graph'], wordset=wordset)
	sort_file(settings.FILES['surface.graph'], key=(1,2), unique=True)
	sort_file(settings.FILES['surface.graph'], key=3)
#	----
	update_file_size(settings.FILES['surface.graph'])
	if settings.LEMMAS_KNOWN:
		filter_lemmas(settings.FILES['surface.graph'])
	filter_rules(settings.FILES['surface.graph'])
	update_file_size(settings.FILES['surface.graph'])
	aggregate_file(settings.FILES['surface.graph'], settings.FILES['surface.rules'], 3)
	rules = RuleSet()
	algorithms.optrules.optimize_rules_in_graph(\
		settings.FILES['training.wordlist'],\
		settings.FILES['surface.graph'],\
		settings.FILES['surface.graph'] + '.opt', rules)
	rename_file(settings.FILES['surface.graph'] + '.opt', settings.FILES['surface.graph'])
	rules.save_to_file('rules.txt.0')

def evaluate():
	print('\nSurface rules: nothing to evaluate.\n')

def import_from_db():
	utils.db.connect()
	print('Importing wordlist...')
	utils.db.pull_table(settings.WORDS_TABLE, ('word', 'freq'),\
		settings.FILES['training.wordlist'])
	print('Importing surface rules...')
	utils.db.pull_table(settings.S_RUL_TABLE, ('rule', 'freq', 'prob'),\
		settings.FILES['surface.rules'])
	# pull graph
	print('Importing graph...')
	with open_to_write(settings.FILES['surface.graph']) as fp:
		for word_1, word_2, rule in utils.db.query_fetch_results('''
			SELECT w1.word, w2.word, r.rule FROM graph g 
				JOIN words w1 ON g.w1_id = w1.w_id
				JOIN words w2 ON g.w2_id = w2.w_id
				JOIN s_rul r ON g.r_id = r.r_id
			;'''):
			write_line(fp, (word_1, word_2, rule))
	# pull surface rules co-occurrences
	print('Importing surface rules co-occurrences...')
	with open_to_write(settings.FILES['surface.rules.cooc']) as fp:
		for rule_1, rule_2, freq, sig in utils.db.query_fetch_results('''
			SELECT r1.rule, r2.rule, c.freq, c.sig FROM s_rul_co c
				JOIN s_rul r1 ON c.r1_id = r1.r_id
				JOIN s_rul r2 ON c.r2_id = r2.r_id
			;'''):
			write_line(fp, (rule_1, rule_2, freq, sig))
	utils.db.close_connection()

def export_to_db():
	# words <- insert ID
	print('Converting wordlist...')
	word_ids = utils.db.insert_id(settings.FILES['training.wordlist'],\
		settings.FILES['wordlist.db'])
	# surface rules <- insert ID
	print('Converting surface rules...')
	s_rule_ids = utils.db.insert_id(settings.FILES['surface.rules'],\
		settings.FILES['surface.rules.db'])
	# graph <- replace words and surface rules with their ID
	print('Converting graph...')
	utils.db.replace_values_with_ids(settings.FILES['surface.graph'],\
		settings.FILES['surface.graph.db'],\
		(word_ids, word_ids, s_rule_ids))
	# surface_rules_cooc <- replace rules with ID
	print('Converting surface rules co-occurrences...')
	utils.db.replace_values_with_ids(settings.FILES['surface.rules.cooc'],\
		settings.FILES['surface.rules.cooc.db'],\
		(s_rule_ids, s_rule_ids, None, None))
	# load tables into DB
	utils.db.connect()
	print('Exporting wordlist...')
	utils.db.push_table(settings.WORDS_TABLE, settings.FILES['wordlist.db'])
	print('Exporting surface rules...')
	utils.db.push_table(settings.S_RUL_TABLE, settings.FILES['surface.rules.db'])
	print('Exporting graph...')
	utils.db.push_table(settings.GRAPH_TABLE, settings.FILES['surface.graph.db'])
	print('Exporting surface rules co-occurrences...')
	utils.db.push_table(settings.S_RUL_CO_TABLE,\
		settings.FILES['surface.rules.cooc.db'])
	utils.db.close_connection()
	# delete temporary files
	remove_file(settings.FILES['wordlist.db'])
	remove_file(settings.FILES['surface.rules.db'])
	remove_file(settings.FILES['surface.graph.db'])
	remove_file(settings.FILES['surface.rules.cooc.db'])

