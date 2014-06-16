from datastruct.counter import *
from utils.files import *
from utils.printer import *
import settings
import math

# TODO na dysku
# TODO co jakis czas sortowanie pliku wynikowego i agregowanie

# mutual information
def minf(freq_12, freq_1, freq_2, total):
	def minf_term(a, b, c):
		if a <= 0 or b*c <= 0: return 0
		return a * math.log(a / (b*c))
	prob_12 = float(freq_12) / total
	prob_1 = float(freq_1) / total
	prob_2 = float(freq_2) / total
	mi = minf_term(prob_12, prob_1, prob_2)
	mi += minf_term(prob_1-prob_12, prob_1, 1-prob_2)
	mi += minf_term(prob_2-prob_12, 1-prob_1, prob_2)
	mi += minf_term(1-prob_1-prob_2+prob_12, 1-prob_1, 1-prob_2)
	return mi

# significance function
SIGNIFICANCE = minf

def calculate_rules_cooc(input_file, output_file, rules_c):
	# load graph and count co-occurrences of rules
	words_count, rule_pairs_count = 0, Counter()
	cur_word, rules = None, []
	pp = progress_printer(get_file_size(input_file))
	print 'Calculating surface rules co-occurrences...'
	for word, edges in load_tsv_file_by_key(input_file, 1):
#		words_count += len(edges)
		words_count += 1
		if len(edges) >= 2:
			for w1, r1 in edges:
				if rules_c.has_key(r1):
					for w2, r2 in edges:
						if rules_c.has_key(r2) and r1 < r2:
							rule_pairs_count.inc((r1, r2))
		for i in range(len(edges)):
			pp.next()
	with open_to_write(output_file) as fp:
		for (r1, r2), count in rule_pairs_count.iteritems():
			sig = SIGNIFICANCE(count, rules_c[r1], rules_c[r2], words_count)
			if sig >= settings.INDEPENDENCY_THRESHOLD:
				write_line(fp, (r1, r2, count, sig))
				
# a generic function for coocs calculation
def calculate_cooc(input_file, output_file):
	# count frequencies of items
	freqs = Counter()
	total_count = 0
	for row in load_tsv_file(input_file):
		freqs.inc(row[1])
		total_count += 1
	# calculate cooccurence significance
	pairs_c = Counter()
	pp = progress_printer(get_file_size(input_file))
	for key, values in load_tsv_file_by_key(input_file, 1):
		values = [v[0] for v in values]
		for v1 in values:
			if freqs.has_key(v1):
				for v2 in values:
					if freqs.has_key(v2) and v1 < v2:
						pairs_c.inc((v1, v2))
		for i in range(len(values)):
			pp.next()
	with open_to_write(output_file) as fp:
		for (v1, v2), count in pairs_c.iteritems():
			sig = SIGNIFICANCE(count, freqs[v1], freqs[v2], total_count)
			if sig >= settings.INDEPENDENCY_THRESHOLD:
				write_line(fp, (v1, v2, count, sig))

