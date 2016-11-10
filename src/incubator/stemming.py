import algorithms.align
from utils.files import *
import re

# "kennenlernen": dlaczego nie odcina -e?

def stem_lexeme(lexeme):
	stems = {}
	minimal_stems = {}
	word_stemmings = {}
	for w in lexeme:
		stems[w] = 1
		word_stemmings[w] = w
	splittings = list(lexeme)
	while len(stems) > 0:
		st, count = max(stems.iteritems(), key = lambda x: len(x[0]))
		del stems[st]
		pair_stems_c = {}
		for st_2, st_2_count in stems.iteritems():
			pair_stem = algorithms.align.lcs(st, st_2)
			if st.find(pair_stem) >= 0:
				if not pair_stems_c.has_key(pair_stem):
					pair_stems_c[pair_stem] = 0
				pair_stems_c[pair_stem] += st_2_count
		if not pair_stems_c:
			minimal_stems[st] = count
			continue
		st_st = max([(x, y) for x, y in pair_stems_c.iteritems() if x != st],\
			key = lambda x: x[1])[0]
		if st_st == st:
			minimal_stems[st] = count
			continue
		if not stems.has_key(st_st):
			stems[st_st] = 0
		stems[st_st] += 1
		splittings_new = []
		for s in splittings:
			p = st.find(st_st)
			replacement = '' if p == 0 else st[:p]+'|'
			replacement += st_st
			replacement += '' if p + len(st_st) >= len(st) else '|'+st[p+len(st_st):]
			splittings_new.append(s.replace(st, replacement))
		for w in word_stemmings.keys():
			if word_stemmings[w] == st:
				word_stemmings[w] = st_st
		splittings = splittings_new
	while len(minimal_stems) > 1:	# handle alternations 
		st, count = max(minimal_stems.iteritems(), key = lambda x: len(x[0]))
		del minimal_stems[st]
		pair_stems_c = {}
		for st_2, st_2_count in minimal_stems.iteritems():
			pair_lcs = algorithms.align.lcs(st, st_2)
			pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in pair_lcs]) + '(.*)')
			m1 = pattern.search(st)
			m2 = pattern.search(st_2)
			pair_stem = ''.join([m2.group(i+1)+pair_lcs[i] for i in range(len(pair_lcs))])
			if not pair_stems_c.has_key(pair_stem):
				pair_stems_c[pair_stem] = 0
			pair_stems_c[pair_stem] += st_2_count
		if not pair_stems_c:
			continue
		st_st = max([(x, y) for x, y in pair_stems_c.iteritems() if x != st],\
			key = lambda x: x[1])[0]
		if not minimal_stems.has_key(st_st):
			minimal_stems[st_st] = 0
		minimal_stems[st_st] += 1

		pair_lcs = algorithms.align.lcs(st, st_st)
		pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in pair_lcs]) + '(.*)')
		m1 = pattern.search(st)
		m2 = pattern.search(st_st)
		replacement = ''
		if m1.group(1):
			replacement += m1.group(1) + '|'
		for i, x in enumerate(m1.groups()[1:-1]):
			replacement += pair_lcs[i]
			if x:
				replacement += '<' + x + '>'
		replacement += pair_lcs[-1]
		if m1.groups()[-1]:
			replacement += '|' + m1.groups()[-1]

		splittings_new = []
		for s in splittings:
			splittings_new.append(s.replace(st, replacement))
		for w in word_stemmings.keys():
			if word_stemmings[w] == st:
				word_stemmings[w] = st_st
		splittings = splittings_new
	return splittings, word_stemmings

def stem_lexeme_lsv(lexeme):
	segmentations = []
	for i in range(len(lexeme)):
		lexeme[i] = '#' + lexeme[i] + '#'
	for word in lexeme:
#		print '\n' + word
		lsv, lpv = [], []
		l = len(word)
		for i in range(l):
#			print set([w[len(w)-i-1] for w in lexeme \
#				if len(w) > i and w.endswith(word[len(word)-i:])])
			lsv.append(len(set([w[i] for w in lexeme \
				if len(w) > i and w.startswith(word[:i])])))
			lpv.append(len(set([w[len(w)-i-1] for w in lexeme \
				if len(w) > i and w.endswith(word[len(word)-i:])])))
		seg = ''
		for i in range(1, l-1):
			if i > 1 and lsv[i] > lsv[i-1] and lsv[i] >= lsv[i+1]:
				seg += '|'
			seg += word[i]
			if i < l-2 and lpv[l-i-1] >= lpv[l-i-2] and lpv[l-i-1] > lpv[l-i]:
				seg += '|'
		seg = seg.replace('||', '|')
		segmentations.append(seg)
	return segmentations

def stem_lexemes(input_file, output_file):
	with open_to_write(output_file) as out_fp:
		for key, lexeme in load_tsv_file_by_key(input_file, 2):
#			splittings, word_st = stem_lexeme([l[0] for l in lexeme])
#			for s in splittings:
#				w = s.replace('|', '').replace('<', '').replace('>', '')
#				write_line(out_fp, (w, s, word_st[w]))
			segmentations = stem_lexeme_lsv([l[0] for l in lexeme])
			for s in segmentations:
				w = s.replace('|', '')
				write_line(out_fp, (w, s))

