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
				write_line(outfp, (rule, float(i_rules_c[rule]) / s_rules_c[rule]))
			else:
				write_line(outfp, (rule, 1.0))

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
		settings.FILES['training.inflection.rules'])
	sort_file(settings.FILES['training.inflection.graph'], key=1)
	algorithms.cooccurrences.calculate_rules_cooc(\
		settings.FILES['training.inflection.graph'],\
		settings.FILES['training.inflection.rules.cooc'], i_rules_c)
