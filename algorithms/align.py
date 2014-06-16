from datastruct.rules import *
import re

# compute the longest common substring of two words
def lcs(word_1, word_2):
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

def extract_rule(word_1, word_2, pattern):
	m1 = pattern.search(word_1)
	m2 = pattern.search(word_2)
	z = zip(m1.groups(), m2.groups())
	for x, y in z:
		if set([l for l in x]) & set([l for l in y]):		# TODO ???
			return None
#			print '\n'+pattern.pattern+'\n', z
#			raise Exception("Sth went wrong: %s, %s" % (word_1, word_2))
	substr = re.sub('\(\.\*\??\)', '', pattern.pattern)
	pref = z[0]
	alt = []
	for i, (x, y) in enumerate(z[1:-1], 1):
		if x or y:
			alt.append((x, y))
	suf = z[-1]
	return Rule(pref, alt, suf)

# extracts the morphological operation needed to turn word_1 into word_2
def align(word_1, word_2):
#	words_lcs = re.sub('([\\.])', '\\\\\\1', lcs(word_1, word_2))		# allow dot in words
	pattern = re.compile('(.*)' + '(.*?)'.join([letter for letter in lcs(word_1, word_2)]) + '(.*)')
	return extract_rule(word_1, word_2, pattern)

