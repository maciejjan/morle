# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import codecs
from algorithms.align import *
from datastruct.rules import *
from utils.files import *

MAX_DATA_SIZE = 100000
TAGS_KNOWN = False
EVAL_FILE = 'maxent.txt'
settings.USE_TAGS = True

def load_data():
	training, testing = [], []
	for word_tag, freq, lemma in read_tsv_file('lexicon.training',\
				print_progress=True, print_msg='Reading training data...'):
		idx = word_tag.rfind('_')
		word, tag = word_tag[:idx], word_tag[idx+1:]
		rule = align(lemma, word_tag).to_string()
		training.append((word, tag, lemma, rule))
	for word_tag, freq, lemma in read_tsv_file('lexicon.eval',\
				print_progress=True, print_msg='Reading testing data...'):
		idx = word_tag.rfind('_')
		word, tag = word_tag[:idx], word_tag[idx+1:]
		rule = align(lemma, word_tag).to_string()
		testing.append((word, tag, lemma, rule))
	training_size = min(len(training), MAX_DATA_SIZE // 2)
	testing_size = min(len(testing), MAX_DATA_SIZE // 2)
	training = training[:training_size]
	testing = testing[:testing_size]
	data = training + testing
	classes = sorted(list(set([rule for word, tag, lemma, rule in data])))
	return data, classes, training_size

def extract_features(data):
	features = []
	for word, tag, lemma, rule in data:
		vec = {}
		for i in range(1, 4):
			vec['^'+str(i)] = word[:i]
			vec[str(i)+'$'] = word[-i:]
			if TAGS_KNOWN:
				vec['tag'] = tag
		features.append(vec)
	vec = DictVectorizer()
	X = vec.fit_transform(features)
	y = [rule for word, tag, lemma, rule in data]
	return X, y

def run():
	data, classes, training_size = load_data()
	X, y = extract_features(data)

	lr = LogisticRegression()
	lr.fit(X[:training_size], y[:training_size])

	# evaluate
	with open_to_write(EVAL_FILE) as evalfp:
		correct_lemmas, correct_tags, correct, total = 0, 0, 0, 0
		for i in range(training_size, len(data)):
			word, tag, lemma, rule = data[i]
			predicted_rule = lr.predict(X[i])[0]
			a = Rule.from_string(predicted_rule).reverse().apply(word+'_'+tag)
			predicted_lemma = a[0] if a else '???'
			# modification begins
#			pred = lr.predict_proba(X[i])[0]
#			predicted_rule, p_max = None, 0.0
#			for i, p in enumerate(pred):
#				if p > p_max:
#					rule_obj = Rule.from_string(classes[i])
#					if rule_obj.rmatch(word+'_'+tag):
#						p_max = p
#						predicted_rule = classes[i]
			# modification ends
#			cl = '+' if rule[:rule.rfind(':')] == predicted_rule[:predicted_rule.rfind(':')] else '-'
			cl = '+' if lemma[:lemma.rfind('_')] == predicted_lemma[:predicted_lemma.rfind('_')] else '-'
			correct_lemmas += 1 if cl == '+' else 0
#			ct = '+' if rule[rule.rfind(':'):] == predicted_rule[predicted_rule.rfind(':'):] else '-'
			ct = '+' if lemma[lemma.rfind('_')+1:] == predicted_lemma[predicted_lemma.rfind('_')+1:] else '-'
			correct_tags += 1 if ct == '+' else 0
			c = '+' if cl == '+' and ct == '+' else '-'
			correct += 1 if c == '+' else 0
#			write_line(evalfp, (word+'_'+tag, predicted_rule, rule, ct, cl))
			write_line(evalfp, (word+'_'+tag, predicted_lemma, lemma, ct, cl))
			total += 1
		write_line(evalfp, ('Lemma:', str(correct_lemmas), str(total), str(float(correct_lemmas)/total)))
		write_line(evalfp, ('Tag:', str(correct_tags), str(total), str(float(correct_tags)/total)))
		write_line(evalfp, ('Lemma+Tag:', str(correct), str(total), str(float(correct)/total)))
	print('Lemma:', correct_lemmas, total, float(correct_lemmas)/total)
	print('Tag:', correct_tags, total, float(correct_tags)/total)
	print('Lemma+Tag:', correct, total, float(correct)/total)

