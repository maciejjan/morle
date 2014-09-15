from datastruct.rules import *
import re

# compute the longest common substring of two words
def lcs(word_1, word_2):
	if settings.USE_TAGS:
		word_1 = word_1[:word_1.rfind(u'_')]
		word_2 = word_2[:word_2.rfind(u'_')]
	previous_row = ['']
	for i in range(0, len(word_1)):
		previous_row.append('')
	for j in range(0, len(word_2)):
		current_row = ['']
		for i in range(0, len(word_1)):
			up = previous_row[i+1]
			diag = previous_row[i] + word_1[i] if word_1[i] == word_2[j] else ''
			left = current_row[-1]
			current_row.append(max([up, diag, left], key = lambda x: len(x)))
		previous_row = current_row
	return previous_row[-1]

# TODO change to use make_rule (currently in algorithms.mdl)
def extract_rule(word_1, word_2, pattern):
	tag = None
	if settings.USE_TAGS:
		p1, p2 = word_1.rfind(u'_'), word_2.rfind(u'_')
		tag = (word_1[p1+1:], word_2[p2+1:])
		word_1, word_2 = word_1[:p1], word_2[:p2]
	m1 = pattern.search(word_1)
	m2 = pattern.search(word_2)
	z = zip(m1.groups(), m2.groups())
	substr = re.sub('\(\.\*\??\)', '', pattern.pattern)
	pref = z[0]
	alt = []
	for i, (x, y) in enumerate(z[1:-1], 1):
		if x or y:
			if x == y:
				return None
			alt.append((x, y))
	suf = z[-1]
	if pref[0] == pref[1] != u'' or suf[0] == suf[1] != u'':
		return None
	return Rule(pref, alt, suf, tag)

# extracts the morphological operation needed to turn word_1 into word_2
def align(word_1, word_2):
#	words_lcs = re.sub('([\\.])', '\\\\\\1', lcs(word_1, word_2))		# allow dot in words
	pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in lcs(word_1, word_2)]) + '(.*)')
	return extract_rule(word_1, word_2, pattern)

