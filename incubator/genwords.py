from datastruct.rules import Rule
from incubator.mdl2 import *
import re

# generating words with MDL model

def make_right_pattern(rule):
	pattern = '^'+rule.prefix[1] + '.*'
	pattern += '.*'.join([y for x, y in rule.alternations])
	pattern += ('.*' if rule.alternations else '') + rule.suffix[1] + '$'
	return re.compile(pattern)

def make_patterns(rules):
	patterns = {}
	for r in rules.values():
		patterns[r.tr] = make_right_pattern(Rule.from_string(r.tr))
	return patterns

def analyze_new_word(rules, lexicon, patterns, word):
	for r, p in patterns.iteritems():
		if p.match(word):
			print r

rules = load_rules('rules.txt.451')
lexicon = load_lexicon('lexicon.txt.451')
patterns = make_patterns(rules)

