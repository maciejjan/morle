# -*- encoding: utf-8 -*-

# load lexemes
# load rule scores
# process graph:
# for each edge -> update the maxima for the given lexemes
# return maxima for each lexeme

# TODO make a list for each lexeme, consider other voting strategies than max. score

from utils.files import *
from utils.printer import *

def ruleprob(S_RUL_PROB_FILE):
	ruleprob_dict = {}
	for row in load_tsv_file(S_RUL_PROB_FILE):
		ruleprob_dict[row[0]] = float(row[1]) / float(row[2])

	def ruleprob_fun(rule):
		return ruleprob_dict[rule] if ruleprob_dict.has_key(rule) else 0.0
	return ruleprob_fun

def load_word_lex(INFLECTION_FILE):
	word_lex = {}
	for word, lex, par in load_tsv_file(INFLECTION_FILE):
		word_lex[word] = lex
	return word_lex

def run():
	word_lex = load_word_lex('inflection.txt')
	ruleprob_fun = ruleprob('s_rul_prob.txt')
	word_max, lex_max = {}, {}
	pp = progress_printer(get_file_size('graph.txt'))
	for word_1, word_2, rule in load_tsv_file('graph.txt'):
		pp.next()
		if word_lex[word_1] == word_lex[word_2]:
			continue
		if not word_max.has_key(word_1) or ruleprob_fun(rule) > word_max[word_1][1]:
			word_max[word_1] = (word_2, ruleprob_fun(rule), rule)
		if word_1 == u'tröstlich':
			print rule, ruleprob_fun(rule)
		if word_lex.has_key(word_1) and word_lex.has_key(word_2):
			lex_1, lex_2 = word_lex[word_1], word_lex[word_2]
			if not lex_max.has_key(lex_1) or ruleprob_fun(rule) > lex_max[lex_1][1]:
				lex_max[lex_1] = (lex_2, ruleprob_fun(rule), rule)
				if word_1 == u'tröstlich':
					print '!'
	with open_to_write('derivation_max.txt') as fp:
		for lex in lex_max.keys():
			if lex_max[lex][1] > 0.1:
				write_line(fp, (lex, lex_max[lex][0], lex_max[lex][1], lex_max[lex][2]))
#		for word, lex in word_lex.iteritems():
#			if word_max.has_key(word) and lex_max.has_key(lex):
#				write_line(fp, (word, word_max[word][0], word_max[word][1], word_max[word][2],\
#					lex, lex_max[lex][0], lex_max[lex][1], lex_max[lex][2]))

