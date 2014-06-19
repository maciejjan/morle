from datastruct.rules import *
import algorithms.ngrams

def rule_score(ruledata, unigrams):
	score = 1.0
	rule = Rule.from_string(ruledata.rule)
	score *= unigrams.word_prob(rule.prefix[0]) / unigrams.word_prob(rule.prefix[1])
	for (x, y) in rule.alternations:
		score *= unigrams.word_prob(x) / unigrams.word_prob(y)
	score *= unigrams.word_prob(rule.suffix[0]) / unigrams.word_prob(rule.suffix[1])
	score *= ruledata.prod
	return score

def prepare_rules_list(rules_file, input_file):
	rules = RuleSet.load_from_file(rules_file)
	del rules[u'#']
	unigrams = algorithms.ngrams.NGramModel(1)
	unigrams.train_from_file(input_file)
	rules_list = [(Rule.from_string(r.rule), rule_score(r, unigrams)) for r in rules.values()]
	rules_list.sort(reverse = True, key = lambda x: x[1])
	return rules_list

def generate_analysis(word, rules_list):
	results = []
	for r, score in rules_list:
		if r.rmatch(word) and score > 1.0:
			results.append((r.to_string(), score))
	return results

