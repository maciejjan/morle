from utils.files import *
import re

class Rule:
	def __init__(self, prefix, alternations, suffix):
		self.prefix = prefix
		self.alternations = alternations
		self.suffix = suffix
		self.left_pattern = None
		self.right_pattern = None
	
	def __eq__(self, other):
		if self.prefix == other.prefix and self.alternations == other.alternations \
			and self.suffix == other.suffix:
			return True
		else:
			return False
	
	def apply(self, word):
		raise Exception('Not implemented yet!')
	
	def get_affixes(self):
		affixes = []
		affixes.append(self.prefix[0])
		affixes.append(self.prefix[1])
		affixes.append(self.suffix[0])
		affixes.append(self.suffix[1])
		return [x for x in affixes if x]
	
	def make_left_pattern(self):
		self.left_pattern = re.compile('^' +\
			self.prefix[0] +\
			('(.*)' if self.alternations else '') + \
			'(.*)'.join([x for x, y in self.alternations]) +\
			'(.*)' + self.suffix[0] + '$')
	
	def make_right_pattern(self):
		self.right_pattern = re.compile('^' +\
			self.prefix[1] +\
			('(.*)' if self.alternations else '') + \
			'(.*)'.join([y for x, y in self.alternations]) +\
			'(.*)' + self.suffix[1] + '$')
	
	def lmatch(self, word):
		if self.left_pattern is None:
			self.make_left_pattern()
		return True if self.left_pattern.match(word) else False
	
	def rmatch(self, word):
		if self.right_pattern is None:
			self.make_right_pattern()
		return True if self.right_pattern.match(word) else False
	
	def reverse(self):
		prefix_r = (self.prefix[1], self.prefix[0])
		alternations_r = [(y, x) for (x, y) in self.alternations]
		suffix_r = (self.suffix[1], self.suffix[0])
		return Rule(prefix_r, alternations_r, suffix_r)

	def to_string(self):
		return self.prefix[0]+':'+self.prefix[1]+'/' +\
			'/'.join([x+':'+y for (x, y) in self.alternations]) +\
			('/' if self.alternations else '') +\
			self.suffix[0]+':'+self.suffix[1]
	
	@staticmethod
	def from_string(string):
		split = string.split('/')
		prefix_sp = split[0].split(':')
		prefix = (prefix_sp[0], prefix_sp[1])
		suffix_sp = split[-1].split(':')
		suffix = (suffix_sp[0], suffix_sp[1])
		alternations = []
		for alt_str in split[1:-1]:
			alt_sp = alt_str.split(':')
			alternations.append((alt_sp[0], alt_sp[1]))
		return Rule(prefix, alternations, suffix)


class RuleData:
	def __init__(self, rule, prod, weight, domsize):
		self.rule = rule
		self.prod = prod
		self.weight = weight
		self.domsize = domsize

class RuleSet:
	def __init__(self):
		self.rules = {}
	
	def __len__(self):
		return len(self.rules)
	
	def __delitem__(self, key):
		del self.rules[key]
	
	def keys(self):
		return self.rules.keys()

	def values(self):
		return self.rules.values()
	
	def has_key(self, key):
		return self.rules.has_key(key)
	
	def __getitem__(self, key):
		return self.rules[key]

	def __setitem__(self, key, val):
		self.rules[key] = val
	
	def logl(self):
		return 0.0
	
	def save_to_file(self, filename):
		with open_to_write(filename) as fp:
			for r_str, r in self.rules.iteritems():
				write_line(fp, (r_str, r.prod, r.weight, r.domsize))
	
	@staticmethod
	def load_from_file(filename):
		rs = RuleSet()
		for rule, prod, weight, domsize in read_tsv_file(filename,\
				types=(unicode, float, float, int)):
			rs[rule] = RuleData(rule, prod, weight, domsize)
		return rs
	
