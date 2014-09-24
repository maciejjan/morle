# -*- encoding: utf-8 -*-

import algorithms.align
import algorithms.fastss
import algorithms.cooccurrences
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
	print 'Loading inflectional rules...'
	for rule, ifreq, freq, weight in read_tsv_file(filename):
		i_rules_c[rule] = weight
	return i_rules_c

### RULE EXTRACTING FUNCTIONS ###

def extract_rules_from_words(words, substring, outfp, comp_outfp, i_rules_c, wordset):
	pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in substring]) + '(.*)')
	for w1, w1_freq in words:
		for w2, w2_freq in words:
			if w1 < w2:
				rule = algorithms.align.extract_rule(w1, w2, pattern)
				if rule is not None and settings.COMPOUNDING_RULES:
					for i in range(3, min(len(w1), len(w2))):
						if i <= len(rule.prefix[0]) and rule.prefix[0][:i] in wordset:
							rr = rule.copy()
							rr.prefix = (u'*' + rule.prefix[0][i:], rule.prefix[1])
							write_line(comp_outfp, (w2, w1, rule.prefix[0][:i], rr.reverse().to_string()))
						if i <= len(rule.prefix[1]) and rule.prefix[1][:i] in wordset:
							rr = rule.copy()
							rr.prefix = (rule.prefix[0], u'*' + rule.prefix[1][i:])
							write_line(comp_outfp, (w1, w2, rule.prefix[1][:i], rr.to_string()))
						if i <= len(rule.suffix[0]) and rule.suffix[0][-i:] in wordset:
							rr = rule.copy()
							rr.suffix = (rule.suffix[0][:-i] + u'*', rule.suffix[1])
							write_line(comp_outfp, (w2, w1, rule.suffix[0][-i:], rr.reverse().to_string()))
						if i <= len(rule.suffix[1]) and rule.suffix[1][-i:] in wordset:
							rr = rule.copy()
							rr.suffix = (rule.suffix[0], rule.suffix[1][:-i] + u'*')
							write_line(comp_outfp, (w1, w2, rule.suffix[1][-i:], rr.to_string()))
				if rule is not None and (i_rules_c is None or i_rules_c.has_key(rule.to_string())):
					if apply_local_filters(rule, ((w1, w1_freq), (w2, w2_freq))):
						write_line(outfp, (w1, w2, rule.to_string()))
						write_line(outfp, (w2, w1, rule.reverse().to_string()))

def extract_rules_from_substrings(input_file, output_file, i_rules_c=None, wordset=None):
	cur_substr, words = '', []
	pp = progress_printer(get_file_size(input_file))
	print 'Extracting rules from substrings...'
	with open_to_write(output_file) as outfp:
		with open_to_write(output_file + '.comp') as comp_outfp:
			for s_len, substr, word, freq in read_tsv_file(input_file):	# TODO _by_key
				if substr != cur_substr:
					if len(words) > 1:
						extract_rules_from_words(words, cur_substr, outfp, comp_outfp, i_rules_c, wordset)
					cur_substr = substr
					words = [(word, int(freq))]
				else:
					words.append((word, int(freq)))
				pp.next()
			if len(words) > 1:
				extract_rules_from_words(words, cur_substr, outfp, i_rules_c)

def filter_and_count_rules(input_file, key=3):
	rules_c = Counter()
	output_file = input_file + '.filtered'
	lines_written = 0
	print 'Filtering and counting rules...'
	with open_to_write(output_file) as outfp:
		pp = progress_printer(get_file_size(input_file))
		for rule_str, wordpairs in read_tsv_file_by_key(input_file, key):
			rule = Rule.from_string(rule_str)
			if apply_global_filters(rule, wordpairs):
				for wp in wordpairs:
					write_line(outfp, tuple(list(wp) + [rule_str]))
					lines_written += 1
#				for (w1, w2) in wordpairs:
#					write_line(outfp, (w1, w2, rule_str))
#					lines_written += 1
				rules_c.inc(rule_str, len(wordpairs))
			for i in range(0, len(wordpairs)):
				pp.next()
	remove_file(input_file)
	rename_file(output_file, input_file)
	set_file_size(input_file, lines_written)
	return rules_c

def filter_new(input_file):
	with open_to_write(input_file + '.fil1') as fp:
		for r, wordpairs in read_tsv_file_by_key(input_file, 3):
			for w1, w2 in wordpairs:
				write_line(fp, (w1, w2, r, len(wordpairs)))
		for r, rows in read_tsv_file_by_key(input_file + '.comp', 4):
			for row in rows:
				write_line(fp, (row[0], row[1], r, len(rows)))
	sort_file(input_file + '.fil1', key=4, reverse=True, numeric=True)
	# build lexicon
	lexicon = Lexicon()
	for w1, w2, r, freq in read_tsv_file(input_file + '.fil1', print_progress=True, print_msg='Building lexicon...'):
		if not lexicon.has_key(w1):
			lexicon[w1] = LexiconNode(w1, 0, 0, 0, 0, 0)
			lexicon.roots.add(w1)
		if not lexicon.has_key(w2):
			lexicon[w2] = LexiconNode(w2, 0, 0, 0, 0, 0)
			lexicon.roots.add(w2)
		if lexicon[w2].prev is None and not w2 in lexicon[w1].analysis():
			lexicon[w2].prev = lexicon[w1]
			lexicon[w1].next[r] = lexicon[w2]
			lexicon.rules_c.inc(r)
			lexicon.rules_c.inc(Rule.from_string(r).reverse().to_string())
			lexicon.roots.remove(w2)
	# filter both graphs
	with open_to_write(input_file + '.fil') as fp:
		for r, wordpairs in read_tsv_file_by_key(input_file, 3):
			if lexicon.rules_c.has_key(r) and lexicon.rules_c[r] > 1:
				for w1, w2 in wordpairs:
					write_line(fp, (w1, w2, r))
			else:
				for w1, w2 in wordpairs:
					if lexicon[w2].prev == lexicon[w1]:
						write_line(fp, (w1, w2, r))
	with open_to_write(input_file + '.comp.fil') as fp:
		for r, rows in read_tsv_file_by_key(input_file + '.comp', 4):
			if lexicon.rules_c.has_key(r) and lexicon.rules_c[r] > 1:
				for w1, w2, w3 in rows:
					if lexicon[w2].prev == lexicon[w1]: # stricter filtering
						write_line(fp, (w1, w2, w3, r))
	remove_file(input_file + '.fil1')
	rename_file(input_file, input_file + '.orig')
	rename_file(input_file + '.fil', input_file)
	rename_file(input_file + '.comp', input_file + '.comp.orig')
	rename_file(input_file + '.comp.fil', input_file + '.comp')
	update_file_size(input_file)
	update_file_size(input_file + '.comp')
#	lexicon.rules_c.save_to_file('s_rul.txt.fil1')
#	lexicon.save_to_file('lex_fil.txt')

### CALCULATING RULE PARAMETERS ###

def join_compounds_to_graph(graph_file, rules):
	sort_file(graph_file + '.comp', key=(3, 4))
	with open_to_write(graph_file, 'a') as fp:
		for k, rows in read_tsv_file_by_key(graph_file + '.comp', (3, 4)):
			new_rule = k[1].replace('*', '*' + k[0] + '*')
			rules[new_rule] = RuleData(new_rule, float(len(rows)) / rules[k[1]].domsize,\
				rules[k[1]].weight, rules[k[1]].domsize)
			for row in rows:
				write_line(fp, (row[0], row[1], new_rule))
	update_file_size(graph_file)

def save_rules(rules, filename):
	print 'Saving rules...'
	with open_to_write(filename) as fp:
		for r in rules:
			write_line(fp, (r.rule, r.prod, r.weight, r.domsize))

def load_wordset(input_file):
	wordset = set([])
	for word, freq in read_tsv_file(input_file):
		wordset.add(word)
	return wordset

### MAIN FUNCTIONS ###

def run():
#	wordset = load_wordset(settings.FILES['training.wordlist'])\
#		if settings.COMPOUNDING_RULES else None
#	algorithms.fastss.create_substrings_file(\
#		settings.FILES['training.wordlist'], settings.FILES['surface.substrings'], wordset)
#	if file_exists(settings.FILES['trained.rules']):
#		extract_rules_from_substrings(settings.FILES['surface.substrings'],\
#			settings.FILES['surface.graph'],\
#			load_training_infl_rules(settings.FILES['trained.rules']))
#	else:
#		extract_rules_from_substrings(settings.FILES['surface.substrings'],\
#			settings.FILES['surface.graph'], wordset=wordset)
#	sort_file(settings.FILES['surface.graph'], key=(1,2), unique=True)
#	sort_file(settings.FILES['surface.graph'], key=3)
##	sort_file(settings.FILES['surface.graph'] + '.comp', key=4)
	update_file_size(settings.FILES['surface.graph'])
##	aggregate_file(settings.FILES['surface.graph'], settings.FILES['surface.rules'], 3)
##	sort_file(settings.FILES['surface.rules'], key=2, reverse=True, numeric=True)
#	sort_file(settings.FILES['surface.graph'] + '.comp', key=(1,3), unique=True)
#	sort_file(settings.FILES['surface.graph'] + '.comp', key=4)
	update_file_size(settings.FILES['surface.graph'] + '.comp')
	filter_new(settings.FILES['surface.graph'])
##	aggregate_file(settings.FILES['surface.graph'] + '.comp', settings.FILES['surface.rules'] + '.comp', 4)
##	sort_file(settings.FILES[surface.rules], key=2, reverse=True, numeric=True)
#	rules_c = filter_and_count_rules(settings.FILES['surface.graph'], 3)
#	rules_c.save_to_file(settings.FILES['surface.rules'])
#	comp_rules_c = filter_and_count_rules(settings.FILES['surface.graph'] + '.comp', 4)
#	comp_rules_c.save_to_file(settings.FILES['surface.rules'] + '.comp')
#	ruleprob = calculate_rule_prob(settings.FILES['wordlist'], rules_c)
#	save_rules(rules_c, ruleprob, settings.FILES['surface.rules'])
#	sort_file(settings.FILES['surface.graph'], key=1)
#	rules_c = Counter.load_from_file(settings.FILES['surface.rules'])
#	if not file_exists(settings.FILES['trained.rules.cooc']):
#		algorithms.cooccurrences.calculate_rules_cooc(\
#			settings.FILES['surface.graph'],\
#			settings.FILES['surface.rules.cooc'], rules_c)
	rules = RuleSet()
	algorithms.optrules.optimize_rules_in_graph(\
		settings.FILES['training.wordlist'],\
		settings.FILES['surface.graph'],\
		settings.FILES['surface.graph'] + '.opt', rules)
	rename_file(settings.FILES['surface.graph'] + '.opt', settings.FILES['surface.graph'])
	algorithms.optrules.calculate_rule_params(\
		settings.FILES['training.wordlist'],\
		settings.FILES['surface.graph'] + '.comp', 4, rules)
	join_compounds_to_graph(settings.FILES['surface.graph'], rules)
	rules.save_to_file('rules.txt.0')
#	rename_file(settings.FILES['surface.graph'] + '.opt', settings.FILES['surface.graph'])

def evaluate():
	print '\nSurface rules: nothing to evaluate.\n'

def import_from_db():
	utils.db.connect()
	print 'Importing wordlist...'
	utils.db.pull_table(settings.WORDS_TABLE, ('word', 'freq'),\
		settings.FILES['training.wordlist'])
	print 'Importing surface rules...'
	utils.db.pull_table(settings.S_RUL_TABLE, ('rule', 'freq', 'prob'),\
		settings.FILES['surface.rules'])
	# pull graph
	print 'Importing graph...'
	with open_to_write(settings.FILES['surface.graph']) as fp:
		for word_1, word_2, rule in utils.db.query_fetch_results('''
			SELECT w1.word, w2.word, r.rule FROM graph g 
				JOIN words w1 ON g.w1_id = w1.w_id
				JOIN words w2 ON g.w2_id = w2.w_id
				JOIN s_rul r ON g.r_id = r.r_id
			;'''):
			write_line(fp, (word_1, word_2, rule))
	# pull surface rules co-occurrences
	print 'Importing surface rules co-occurrences...'
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
	print 'Converting wordlist...'
	word_ids = utils.db.insert_id(settings.FILES['training.wordlist'],\
		settings.FILES['wordlist.db'])
	# surface rules <- insert ID
	print 'Converting surface rules...'
	s_rule_ids = utils.db.insert_id(settings.FILES['surface.rules'],\
		settings.FILES['surface.rules.db'])
	# graph <- replace words and surface rules with their ID
	print 'Converting graph...'
	utils.db.replace_values_with_ids(settings.FILES['surface.graph'],\
		settings.FILES['surface.graph.db'],\
		(word_ids, word_ids, s_rule_ids))
	# surface_rules_cooc <- replace rules with ID
	print 'Converting surface rules co-occurrences...'
	utils.db.replace_values_with_ids(settings.FILES['surface.rules.cooc'],\
		settings.FILES['surface.rules.cooc.db'],\
		(s_rule_ids, s_rule_ids, None, None))
	# load tables into DB
	utils.db.connect()
	print 'Exporting wordlist...'
	utils.db.push_table(settings.WORDS_TABLE, settings.FILES['wordlist.db'])
	print 'Exporting surface rules...'
	utils.db.push_table(settings.S_RUL_TABLE, settings.FILES['surface.rules.db'])
	print 'Exporting graph...'
	utils.db.push_table(settings.GRAPH_TABLE, settings.FILES['surface.graph.db'])
	print 'Exporting surface rules co-occurrences...'
	utils.db.push_table(settings.S_RUL_CO_TABLE,\
		settings.FILES['surface.rules.cooc.db'])
	utils.db.close_connection()
	# delete temporary files
	remove_file(settings.FILES['wordlist.db'])
	remove_file(settings.FILES['surface.rules.db'])
	remove_file(settings.FILES['surface.graph.db'])
	remove_file(settings.FILES['surface.rules.cooc.db'])

