from datastruct.rules import *
import libhfst
from operator import itemgetter

MAX_INFIX_LENGTH = 3
MAX_AFFIX_LENGTH = 5
INFIX_SLOTS = 1

def align_words(word_1, word_2):
	previous_row = [(0, ())]
	for i in range(len(word_1)):
		left_dist, left_seq = previous_row[-1]
		previous_row.append((left_dist + 1,\
		                     left_seq + ((word_1[i], libhfst.EPSILON),)))
	for j in range(len(word_2)):
		up_dist, up_seq = previous_row[0]
		current_row = [(up_dist + 1, up_seq + ((libhfst.EPSILON, word_2[j]),))]
		for i in range(len(word_1)):
			up_dist, up_seq = previous_row[i+1]
			diag_dist, diag_seq = previous_row[i]
			left_dist, left_seq = current_row[-1]
			# in case of ties, prefer the diagonal
			current_row.append(min((
				(diag_dist + int(word_1[i]!=word_2[j]),
					diag_seq + ((word_1[i], word_2[j]),)),
				(up_dist + 1, up_seq + ((libhfst.EPSILON, word_2[j]),)),
				(left_dist + 1, left_seq + ((word_1[i], libhfst.EPSILON),))),
				key = itemgetter(0)))
		previous_row = current_row
	return previous_row[-1][1]

def extract_rule(node_1, node_2):
	'''Extract the most general rule.'''
	alignment = align_words(node_1.word, node_2.word)
	seq = []
	for x, y in alignment:
		if x == y:
			if not seq or seq[-1] != (libhfst.IDENTITY, libhfst.IDENTITY):
				seq.append((libhfst.IDENTITY, libhfst.IDENTITY))
		else:
			seq.append((x, y))
	rule = Rule.from_seq(seq, (node_1.tag, node_2.tag))
	# check whether the rule conforms to the criteria
	if rule.check(MAX_AFFIX_LENGTH, MAX_INFIX_LENGTH, INFIX_SLOTS) and\
			rule.reverse().check(MAX_AFFIX_LENGTH, MAX_INFIX_LENGTH, INFIX_SLOTS):
		return rule
	else:
		return None

def extract_all_rules(node_1, node_2):
	'''Extract all rules fitting the pair of words.'''
	# queue: rule seq, remaining alignment, num_segments, length of the last segment, is_last_segment
	alignment = align_words(node_1.word, node_2.word)
#	print(alignment)
	queue = [((), alignment, 1, 0, False)]
	results = []
	while queue:
		seq, alignment, num_seg, len_seg, last_seg = queue.pop()
		if not alignment:
			results.append(Rule.from_seq(seq, (node_1.tag, node_2.tag)))
			continue
		x, y = alignment[0]
		if x == y:
			# prolong the prefix
			if num_seg == 1 and len_seg < MAX_AFFIX_LENGTH:
				queue.append((seq + ((x, y),),
				              alignment[1:], num_seg, len_seg+1, last_seg))
			# prolong the suffix
			if num_seg > 1 and len_seg < MAX_AFFIX_LENGTH:
				queue.append((seq + ((x, y),),
				              alignment[1:], num_seg, len_seg+1, True))
			if not last_seg:
				# prolong the current alternation
				if num_seg > 1 and not last_seg and len_seg < MAX_INFIX_LENGTH:
					if seq[-1] != (libhfst.IDENTITY, libhfst.IDENTITY):
						queue.append((seq + ((x, y),),
									  alignment[1:], num_seg, len_seg+1, last_seg))
				# insert or continue an identity symbol
				if not seq or seq[-1] != (libhfst.IDENTITY, libhfst.IDENTITY):
					if num_seg < INFIX_SLOTS+2:
						queue.append((seq + ((libhfst.IDENTITY, libhfst.IDENTITY),),
									  alignment[1:], num_seg+1, 0, last_seg))
				else:
					queue.append((seq, alignment[1:],\
								  num_seg, len_seg, last_seg))
		else:
			queue.append((seq + ((x, y),), alignment[1:],\
			              num_seg, len_seg+1, last_seg))
	return list(set(results))

