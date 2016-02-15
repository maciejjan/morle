from utils.files import *
from utils.printer import *
import re

MAX_DIST = 10
MIN_LENGTH = 2
MIN_COMP_LEN = 5
MAX_ALTERNATION_LENGTH = 3
MAX_AFFIX_LENGTH = 5
INFIX_SLOTS = 2
MAX_OUTPUT_SIZE = 5e10	# 50 GB

# turns a zero into one at the chosen position in a string of zeros and ones
def zero_to_one(old_mask, pos):
	new_mask = ''
	current_pos = 0
	for i in range(0, len(old_mask)):
		if current_pos == pos:
			new_mask += '1'
		else:
			new_mask += old_mask[i]
		if old_mask[i] == '0':
			current_pos += 1
	return new_mask

def mask_correct(mask):
	mask_split = re.split('0+', mask)
	if len(mask_split) > INFIX_SLOTS+2:
		return False
	if len(mask_split[0]) > MAX_AFFIX_LENGTH:
		return False
	if len(mask_split[-1]) > MAX_AFFIX_LENGTH:
		return False
	if len(mask_split) <= 2: # no alternations
		return True
	for x in mask_split[1:-1]:
		if len(x) > MAX_ALTERNATION_LENGTH:
			return False
	return True

# get all substrings of a given word matching the correctness conditions
def substrings_for_word(word, wordset):
	if settings.USE_TAGS:
		word = word[:word.rfind('_')]
	substrings = [[(word, '0' * len(word))]]
	for i in range(0, min(len(word)//2, MAX_DIST, len(word)-MIN_LENGTH)):
		new_substrings = set([])
		for s, mask in substrings[i]:
			for j in range(0, len(s)):
				new_substring = s[:j] + s[j+1:]
				new_mask = zero_to_one(mask, j)
				if mask_correct(new_mask):
					new_substrings.add((new_substring, new_mask))
		substrings.append(new_substrings)
	result = []
	for substring_set in substrings:
		result.extend([s for s, mask in substring_set])
	# generate substrings for compounding rules
	if settings.COMPOUNDING_RULES:
		for i in range(1, len(word)-1):
			if word[i:] in wordset and len(word[i:]) >= MIN_COMP_LEN:
				result.append(word[i:])
			if word[:-i] in wordset and len(word[i:]) >= MIN_COMP_LEN:
				result.append(word[:-i])
			# for words starting with capitalization mark
			if '"' + word[i:] in wordset and len(word[i:]) >= MIN_COMP_LEN:
				result.append(word[i:])
			if '"' + word[:-i] in wordset and len(word[i:]) >= MIN_COMP_LEN:
				result.append(word[:-i])
	# remove duplicates -- same substrings generated with different masks
	return list(set(result))

# generate substrings for all words
def generate_substrings(input_file, output_file, wordset):
	with open_to_write(output_file) as fp:
		bytes_written = 0
		lines_written = 0
		print('Generating substrings...')
		pp = progress_printer(get_file_size(input_file))
		for word, freq in read_tsv_file(input_file):
			for s in substrings_for_word(word, wordset):
				bytes_written += write_line(fp, (len(s), s, word, freq),
					count_bytes=True)
				lines_written += 1
			if bytes_written > MAX_OUTPUT_SIZE:
				raise Exception("FastSS: maximum output size exceeded!")
			next(pp)
	set_file_size(output_file, lines_written)

def create_substrings_file(input_file, substrings_file, wordset):
	generate_substrings(input_file, substrings_file, wordset)
	sort_file(substrings_file, key=2)
	sort_file(substrings_file, key=1, numeric=True, reverse=True, stable=True)
#	sort_file(substrings_file, reverse=True, numeric=True)	# TODO key - umlaut sorting ok?

