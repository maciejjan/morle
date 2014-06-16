class Rule:
	def __init__(self, prefix, alternations, suffix):
		self.prefix = prefix
		self.alternations = alternations
		self.suffix = suffix
	
	def __eq__(self, other):
		if self.prefix == other.prefix and self.alternations == other.alternations \
			and self.suffix == other.suffix:
			return True
		else:
			return False
	
	def get_affixes(self):
		affixes = []
		affixes.append(self.prefix[0])
		affixes.append(self.prefix[1])
		affixes.append(self.suffix[0])
		affixes.append(self.suffix[1])
		return [x for x in affixes if x]
	
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

class OperationSet:
	def __init__(self, ops = None):
		if ops is None:
			self.opset = set([])
		else:
			self.opset = set(ops)
	
	def add(self, op):
		self.opset.add(op)
	
	def __contains__(self, op):
		return op in self.opset
	
	def __eq__(self, other):
		return self.opset == other.opset
	
	def __len__(self):
		return len(self.opset)
	
	def __and__(self, other):
		result = OperationSet()
		result.opset = self.opset & other.opset
		return result
	
	def __or__(self, other):
		result = OperationSet()
		result.opset = self.opset | other.opset
		return result

	def __iand__(self, other):
		self.opset &= other.opset
		return self

	def __ior__(self, other):
		self.opset |= other.opset
		return self

	def iterstrings(self):
		for op_str in self.opset:
			yield op_str
	
	def to_string(self):
		return ', '.join(sorted(list(self.opset)))
	
	@staticmethod
	def from_string(string):
		result = OperationSet()
		split = string.split(', ')
		for op_str in split:
			result.add(op_str)
		return result

class OperationMultiSet:
	def __init__(self):
		self.opsets = set([])
	
	def add(self, opset):
		self.opsets.add(opset.to_string())
	
	def __iter__(self):
		for opset_str in self.opsets:
			yield opset_str
