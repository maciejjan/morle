DICT = 1
TRIE = 2
LIST = 3

def new_storage_struct(struct):
	if struct == DICT:
		return {}
	elif struct == TRIE:
		return Trie()
	elif struct == LIST:
		return ListStorage()
	else:
		raise Exception("Unknown storage structure: " + str(struct))

# TODO TupleTrie: tuples as keys?

class Trie:
	def __init__(self, key = ''):
		self.key = key
		self.value = None
		self.children = {}
	
	def __del__(self):
		del self.children
	
	def __delitem__(self, key):
		if len(key) == 0:
			self.__del__()
		elif len(key) == 1:
			del self.children[key]
		else:
			self.children[key[0]].__delitem__(key[1:])
		# TODO delete branches that don't contain any key
	
	def __getitem__(self, key):
		if len(key) == 0:
			return self.value
		else:
			return self.children[key[0]].__getitem__(key[1:])

	def __len__(self):
		result = 0
		if self.value is not None:
			result += 1
		for c in self.children.values():
			result += len(c)
		return result
	
	def __setitem__(self, key, val):
		if len(key) == 0:
			self.value = val
		else:
			if not self.children.has_key(key[0]):
				self.children[key[0]] = Trie(self.key + key[0])
			self.children[key[0]].__setitem__(key[1:], val)

	def has_key(self, key):
		if len(key) == 0:
			return self.value is not None
		else:
			if not self.children.has_key(key[0]):
				return False
			return self.children[key[0]].has_key(key[1:])
	
	def iteritems(self):
		if self.value is not None:
			yield self.key, self.value
		for c in self.children.values():
			for key, value in c.iteritems():
				yield key, value
	
	def keys(self):
		results = []
		if self.value is not None:
			results.append(self.key)
		for c in self.children.values():
			results.extend(c.keys())
		return results
	
	def values(self):
		results = []
		if self.value is not None:
			results.append(self.value)
		for c in self.children.values():
			results.extend(c.values())
		return results

class ListStorage:
	def __init__(self):
		self.data = []
	
	def __del__(self):
		del self.data
	
	def __delitem__(self, key):
		for (k, v) in self.data:
			if k == key:
				self.data.remove((k, v))
	
	def __getitem__(self, key):
		for (k, v) in self.data:
			if k == key:
				return v

	def __len__(self):
		return len(self.data)
	
	def __setitem__(self, key, val):
		self.__delitem__(key)
		self.data.append((key, val))

	def has_key(self, key):
		for (k, v) in self.data:
			if k == key:
				return True
		return False
	
	def iteritems(self):
		for (k, v) in self.data:
			yield (k, v)
	
	def keys(self):
		return [k for (k, v) in self.data]
	
	def values(self):
		return [v for (k, v) in self.data]

