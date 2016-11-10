# extract paradigms from lexemes

import algorithms.align
from utils.files import *

WORDLIST_FILE = 'input.txt'
INPUT_FILE = 'gs_inflection.txt.sorted'
OUTPUT_FILE = 'inflection.txt.sorted'
INPUT_LOG_FILE = 'log1.txt'
OUTPUT_LOG_FILE = 'log2.txt'

def extract_paradigm(lexeme, word):
	paradigm = []
	for word_2 in lexeme:
		if word_2 != word:
			rule = algorithms.align.align(word, word_2)
			paradigm.append(rule.to_string())
	return '-' if not paradigm else ','.join(sorted(paradigm))

def load_wordlist(filename):
	wordlist = set([])
	for word, freq in load_tsv_file(filename):
		wordlist.add(word)
	return wordlist

def load_clusters(filename, wordlist):
	clusters = {}
	for label, rows in load_tsv_file_by_key(filename, 3):
		clusters[label] = set([r[0] for r in rows if r[0] in wordlist])
	return clusters

def cluster_purity(clusters, classes, log_file):
	n = sum([len(c) for c in clusters.values()])
	purity = 0
	with open_to_write(log_file) as fp:
		for clust_lab, cluster in clusters.iteritems():
			match, match_lab, matched = 0.0, '', set([])
			for cls_lab, cls in classes.iteritems():
#			match = max(match, len(cluster & cls))
				if len(cluster & cls) > match:
					match = len(cluster & cls)
					match_lab = cls_lab
					matched = cluster & cls
			purity += match
			write_line(fp, (clust_lab + ' -> ' + match_lab, ','.join(list(matched))))
#		if match < len(cluster):
#		print match, len(cluster)
	return float(purity) / n

def run():
	with open_to_write(OUTPUT_FILE) as outfp:
		for base, rows in load_tsv_file_by_key(INPUT_FILE, 2):
			lexeme = set([r[0] for r in rows])
			for word in lexeme:
				paradigm = extract_paradigm(lexeme, word)
				write_line(outfp, (word, base, paradigm))
	sort_file(OUTPUT_FILE, key=(3,3))

def evaluate():
	wordlist = load_wordlist(WORDLIST_FILE)
	clusters = load_clusters(OUTPUT_FILE, wordlist)
	classes = load_clusters(INPUT_FILE, wordlist)
	purity = cluster_purity(clusters, classes, INPUT_LOG_FILE)
	inv_purity = cluster_purity(classes, clusters, OUTPUT_LOG_FILE)
	print ''
	print 'Purity:', purity
	print 'Inverse Purity:', inv_purity
	print ''

