from utils.files import *
from utils.printer import *
from collections import defaultdict
import re

MAX_DIST = 10
MIN_LENGTH = 2
MIN_COMP_LEN = 5
MAX_INFIX_LENGTH = 3
MAX_AFFIX_LENGTH = 5
INFIX_SLOTS = 1
MAX_OUTPUT_SIZE = 5e10	# 50 GB

## turns a zero into one at the chosen position in a string of zeros and ones
#def zero_to_one(old_mask, pos):
#	new_mask = ''
#	current_pos = 0
#	for i in range(0, len(old_mask)):
#		if current_pos == pos:
#			new_mask += '1'
#		else:
#			new_mask += old_mask[i]
#		if old_mask[i] == '0':
#			current_pos += 1
#	return new_mask
#
#def mask_correct(mask):
#	mask_split = re.split('0+', mask)
#	if len(mask_split) > INFIX_SLOTS+2:
#		return False
#	if len(mask_split[0]) > MAX_AFFIX_LENGTH:
#		return False
#	if len(mask_split[-1]) > MAX_AFFIX_LENGTH:
#		return False
#	if len(mask_split) <= 2: # no alternations
#		return True
#	for x in mask_split[1:-1]:
#		if len(x) > MAX_ALTERNATION_LENGTH:
#			return False
#	return True
#
## get all substrings of a given word matching the correctness conditions
#def substrings_for_word(word, wordset):
#	if settings.USE_TAGS:
#		word = word[:word.rfind('_')]
#	substrings = [[(word, '0' * len(word))]]
#	for i in range(0, min(len(word)//2, MAX_DIST, len(word)-MIN_LENGTH)):
#		new_substrings = set([])
#		for s, mask in substrings[i]:
#			for j in range(0, len(s)):
#				new_substring = s[:j] + s[j+1:]
#				new_mask = zero_to_one(mask, j)
#				if mask_correct(new_mask):
#					new_substrings.add((new_substring, new_mask))
#		substrings.append(new_substrings)
#	result = []
#	for substring_set in substrings:
#		result.extend([s for s, mask in substring_set])
#	# generate substrings for compounding rules
#	if settings.COMPOUNDING_RULES:
#		for i in range(1, len(word)-1):
#			if word[i:] in wordset and len(word[i:]) >= MIN_COMP_LEN:
#				result.append(word[i:])
#			if word[:-i] in wordset and len(word[i:]) >= MIN_COMP_LEN:
#				result.append(word[:-i])
#			# for words starting with capitalization mark
#			if '"' + word[i:] in wordset and len(word[i:]) >= MIN_COMP_LEN:
#				result.append(word[i:])
#			if '"' + word[:-i] in wordset and len(word[i:]) >= MIN_COMP_LEN:
#				result.append(word[:-i])
#	# remove duplicates -- same substrings generated with different masks
#	return list(set(result))

# TODO compounds

def slice_word(word, slices):
	results = []
	queue = [(word, slices)]
	visited = set()
	while queue:
		word, slices = queue.pop()
		if slices in visited: continue
		visited.add(slices)
#		print(slices)
#		results.append(sum((word[i:j] for i, j in slices), ()))
		substr = sum((word[i:j] for i, j in slices), ())
#		substr = ''.join(sum((word[i:j] for i, j in slices), ()))
		results.append(''.join(substr))
		if len(substr) <= len(word)/2: continue
		for i, (begin, end) in enumerate(slices):
			if begin < end-1:
				if i == 0:
					if begin < MAX_AFFIX_LENGTH:
						queue.append((word, ((begin+1, end),) + slices[1:]))
				elif begin-slices[i-1][1]+1 < MAX_INFIX_LENGTH:
					queue.append(\
						(word,
						 slices[:i] + ((begin+1, end),) + slices[i+1:]))
				if i == len(slices)-1:
					if len(word)-end+1 < MAX_AFFIX_LENGTH:
						queue.append((word, slices[:i] + ((begin, end-1),)))
				elif slices[i+1][0]-end+1 < MAX_INFIX_LENGTH:
					queue.append(\
						(word,
						 slices[:i] + ((begin, end-1),) + slices[i+1:]))
			if len(slices) < INFIX_SLOTS + 1:
				for mid in range(begin+1, end-1):
					queue.append((\
						word,
						slices[:i] +\
							((begin, mid), (mid+1, end)) +\
							slices[i+1:]))
	return results

def substrings_for_word(word):
	return set(slice_word(word, ((0, len(word)),)))

# generate substrings for all words
#def generate_substrings(input_file, output_file, wordset):
#	with open_to_write(output_file) as fp:
#		bytes_written = 0
#		lines_written = 0
#		print('Generating substrings...')
#		pp = progress_printer(get_file_size(input_file))
#		for (word,) in read_tsv_file(input_file, (str,)):
#			for s in substrings_for_word(word, wordset):
#				bytes_written += write_line(fp, (len(s), s, word),
#					count_bytes=True)
#				lines_written += 1
#			if bytes_written > MAX_OUTPUT_SIZE:
#				raise Exception("FastSS: maximum output size exceeded!")
#			next(pp)
#	set_file_size(output_file, lines_written)
#
#def create_substrings_file(input_file, substrings_file, wordset):
#	generate_substrings(input_file, substrings_file, wordset)
#	sort_file(substrings_file, key=2)
#	sort_file(substrings_file, key=1, numeric=True, reverse=True, stable=True)
##	sort_file(substrings_file, reverse=True, numeric=True)	# TODO key - umlaut sorting ok?
#

def generate_substrings(lexicon, print_progress=False):
	pp = progress_printer(len(lexicon)) if print_progress else None
	for node in lexicon.values():
		for substr in substrings_for_word(node.word):
			yield (substr, node)
		if print_progress:
			next(pp)

def create_substrings_file(lexicon, output_file):
	with open_to_write(output_file) as fp:
		for substr, node in generate_substrings(lexicon, print_progress=True):
			write_line(fp, (substr, ''.join(node.word)))

def create_substrings_hash(lexicon):
	result = defaultdict(lambda: list())
	for substr, node in generate_substrings(lexicon, print_progress=True):
		result[substr].append(node)
	return result

def similar_words(lexicon, print_progress=False):
	substr_hash = defaultdict(lambda: list())
	if print_progress:
		pp = progress_printer(len(lexicon))
	for node in lexicon.values():
		sim_nodes = set()
		for substr in substrings_for_word(node.word):
			for node2 in substr_hash[substr]:
				sim_nodes.add(node2)
			substr_hash[substr].append(node)
		for node2 in sim_nodes:
			yield node, node2
		if print_progress:
			next(pp)

