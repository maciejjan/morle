from algorithms.cooccurrences import minf
from datastruct.counter import Counter
from files import *
import math

SIG_THRESHOLD = 6.0
COOC_FILE = 'cooc.txt'
LEX_COOC_FILE = 'lex_cooc.txt'
#INV_W_FILE = 'inv_w.txt'
#INV_W_FILE = '../../../data/pol_mixed_2013/pol_mixed_2013.inv_w.txt'
INV_W_FILE = '../../../data/deu_wikipedia_2010/deu_wikipedia_2010.inv_w.txt'
INFLECTION_FILE = 'inflection.txt'
PAR_COOC_FILE = 'par_cooc.txt'
NEIGHBOURHOOD = True

def entropy(val_counts):
	total = sum(val_counts.values())
	entr = 0.0
	for count in val_counts.values():
		p = float(count) / total
		entr -= p * math.log(p)
	return entr

word_lex, word_par = {}, {}
for word, lex, par in load_tsv_file(INFLECTION_FILE):
	word_lex[word] = lex
	word_par[word] = par

lex_freq, lex_total = {}, 0
par_freq, par_total = {}, 0

with open_to_write(COOC_FILE) as fp:
	for s_id, words in load_tsv_file_by_key(INV_W_FILE, 3):
		words = [(w.lower(), pos) for (w, pos) in words\
			if word_lex.has_key(w.lower()) and word_par.has_key(w.lower())]
		for (w, pos) in words:
			if not lex_freq.has_key(word_lex[w]):
				lex_freq[word_lex[w]] = 0
			lex_freq[word_lex[w]] += 1
			lex_total += 1
			if not par_freq.has_key(word_par[w]):
				par_freq[word_par[w]] = 0
			par_freq[word_par[w]] += 1
			par_total += 1
		if len(words) > 1:
			if NEIGHBOURHOOD:
				for (w1, pos1) in words:
					for (w2, pos2) in words:
						if int(pos1)+1 == int(pos2):
							write_line(fp, (w1, w2, word_lex[w1], word_lex[w2],\
								word_par[w1], word_par[w2], s_id))
			else:
				for i in range(len(words)):
					for j in range(i) + range(i+1, len(words)):
						write_line(fp, (words[i][0], words[j][0], word_lex[words[i][0]],\
							word_lex[words[j][0]], word_par[words[i][0]], word_par[words[j][0]], s_id))

# sort by lexemes
sort_file(COOC_FILE, key=(3, 4))
with open_to_write(LEX_COOC_FILE) as fp:
	for (lex_1, lex_2), words in load_tsv_file_by_key(COOC_FILE, (3, 4)):
		# calculate significance
		sig = 2 * lex_total * minf(len(words), lex_freq[lex_1], lex_freq[lex_2], lex_total)
		if sig > SIG_THRESHOLD:
			par_1, par_2, par_12 = Counter(), Counter(), Counter()
			for w in words:
				par_1.inc(w[2])
				par_2.inc(w[3])
				par_12.inc((w[2], w[3]))
			num_par_1 = len(par_1)
			num_par_2 = len(par_2)
			num_par_12 = len(par_12)
			entr_par_1 = entropy(par_1)
			entr_par_2 = entropy(par_2)
			entr_par_12 = entropy(par_12)
			par_mi = entr_par_1 + entr_par_2 - entr_par_12
			dep = 0.0 if entr_par_12 == 0.0 else par_mi / entr_par_12
			write_line(fp, (lex_1, lex_2, len(words), sig, num_par_1, num_par_2,\
				entr_par_1, entr_par_2, entr_par_12, par_mi, dep))

# sort by paradigms
sort_file(COOC_FILE, key=(5, 6))
with open_to_write(PAR_COOC_FILE) as fp:
	for (par_1, par_2), words in load_tsv_file_by_key(COOC_FILE, (5, 6)):
		# calculate significance
		sig = 2 * par_total * minf(len(words), par_freq[par_1], par_freq[par_2], par_total)
		if sig > SIG_THRESHOLD:
			lex_1, lex_2, lex_12 = Counter(), Counter(), Counter()
			for w in words:
				lex_1.inc(w[2])
				lex_2.inc(w[3])
				lex_12.inc((w[2], w[3]))
			num_lex_1 = len(lex_1)
			num_lex_2 = len(lex_2)
			num_lex_12 = len(lex_12)
			entr_lex_1 = entropy(lex_1)
			entr_lex_2 = entropy(lex_2)
			entr_lex_12 = entropy(lex_12)
			lex_mi = entr_lex_1 + entr_lex_2 - entr_lex_12
			dep = 0.0 if entr_lex_12 == 0.0 else lex_mi / entr_lex_12
			write_line(fp, (par_1, par_2, len(words), sig, num_lex_1, num_lex_2,\
				entr_lex_1, entr_lex_2, entr_lex_12, lex_mi, dep))
