from utils.db import *
from utils.files import *

def segment_word():
	pass

def segment_wordset(wordset):
	segmentations = []
	for i in range(len(wordset)):
		wordset[i] = '#' + wordset[i] + '#'
	for word in wordset:
#		print '\n' + word
		lsv, lpv = [], []
		l = len(word)
		for i in range(l):
			if word == u'#altersteilzeitarbeit#':
				print i, set([w[len(w)-i-1] for w in wordset \
					if len(w) > i and w.endswith(word[len(word)-i:])])
			lsv.append(len(set([w[i] for w in wordset \
				if len(w) > i and w.startswith(word[:i])])))
			lpv.append(len(set([w[len(w)-i-1] for w in wordset \
				if len(w) > i and w.endswith(word[len(word)-i:])])))
		if word == u'#altersteilzeitarbeit#':
			print word
			print wordset
			print lsv
			print lpv
		seg = ''
		for i in range(1, l-1):
			if i > 1 and lsv[i] > lsv[i-1] and lsv[i] >= lsv[i+1]:
				seg += '|'
			seg += word[i]
			if i < l-2 and lpv[l-i-1] >= lpv[l-i-2] and lpv[l-i-1] > lpv[l-i]:
				seg += '|'
		seg = seg.replace('||', '|')
		if word == u'#altersteilzeitarbeit#':
			print seg
		segmentations.append(seg)
	return segmentations

def find_stem():
	pass

def run():
	print 'Segmenting lexemes...'
	# segment lexemes 
	with open_to_write(settings.FILES['segmentation']) as fp:
		pp = progress_printer(get_file_size(settings.FILES['inflection']))
		for base, rows in load_tsv_file_by_key(settings.FILES['inflection'], 2):
			lexeme = [r[0] for r in rows]
			segmentations = segment_wordset(lexeme)
			for seg in segmentations:
				write_line(fp, (seg.replace('|', ''), seg))
				pp.next()
	# find stems of lexemes
	# load derivations
	# for each stem -- segment its "derivational family"
	# find root -- stem of the derivational family
	pass

def evaluate():
	pass

def import_from_db():
	pass

def export_to_db():
	pass
