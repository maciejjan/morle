from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge
from datastruct.rules import Rule

from collections import defaultdict
import numpy as np
from operator import itemgetter
from typing import Dict, Iterable, List, Tuple

import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 10

# TODO
# - move model into separate class: NeuralModel
# - test on real data and verify the convergence


def extract_n_grams(word :Iterable[str]) -> Iterable[Iterable[str]]:
    result = []
    max_n = min(MAX_NGRAM_LENGTH, len(word))+1
    result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
    result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
    return result


def select_ngram_features(edges :List[GraphEdge]) -> List[str]:
    # count n-gram frequencies
    ngram_freqs = defaultdict(lambda: 0)
    for edge in edges:
        for ngram in extract_n_grams(edge.source.symstr):
            ngram_freqs[ngram] += 1
    # select most common n-grams
    ngram_features = \
        list(map(itemgetter(0), 
                 sorted(ngram_freqs.items(), 
                        reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
    return ngram_features


def extract_features_from_edges(edges, features, rule_idx):
    attributes, rule_ids = [], []
    for edge in edges:
        attribute_vec = []
        edge_ngrams = set(extract_n_grams(edge.source.symstr))
        for ngram in features:
            attribute_vec.append(1 if ngram in edge_ngrams else 0)
        attributes.append(attribute_vec)
        rule_ids.append([rule_idx[edge.rule]])
    return np.array(attributes), np.array(rule_ids)


def prepare_model(num_features, num_rules):
    print('num_features = {}'.format(num_features))
    print('num_rules = {}'.format(num_rules))
    input_features = Input(shape=(num_features,), name='input_features')
    input_rule = Input(shape=(1,), name='input_rule')
    rule_emb = Embedding(input_dim=num_rules, output_dim=100, input_length=1)(input_rule)
    rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
    concat = keras.layers.concatenate([input_features, rule_emb_fl])
    dense = Dense(1000, activation='softplus', name='dense')(concat)
    output = Dense(1, activation='sigmoid', name='output')(dense)

    model = Model(inputs=[input_features, input_rule], outputs=[output])
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


def run():
    words_raw = ['machen', 'gemacht', 'macht', 'machte', 'gemachte']
    rules_raw = [':ge/en:t', ':/en:t', ':/:e', ':/en:te', 'ge:/:']
    words = [LexiconEntry(w) for w in words_raw]
    rules = [Rule.from_string(r) for r in rules_raw]
    edges = [
        GraphEdge(words[0], words[1], rules[0]),
        GraphEdge(words[0], words[2], rules[1]),
        GraphEdge(words[1], words[4], rules[2]),
        GraphEdge(words[2], words[3], rules[2]),
        GraphEdge(words[0], words[3], rules[3]),
        GraphEdge(words[1], words[2], rules[4]),
        GraphEdge(words[4], words[3], rules[4]),
    ]
    for edge in edges:
        print(str(edge.source), str(edge.target), str(edge.rule), sep='\t')
    y = np.array([1, 1, 1, 0.3, 0.7, 0.05, 0.05])

    features = select_ngram_features(edges)
    rule_idx = { rule: idx for idx, rule in enumerate(set(edge.rule for edge in edges)) }
    X_attr, X_rule = extract_features_from_edges(edges, features, rule_idx)
    print(features)
    for rule, idx in rule_idx.items():
        print(rule, idx)
    print(X_attr)
    print(X_rule)
    print(y)
    model = prepare_model(X_attr.shape[1], len(rule_idx))
    print(model.predict([X_attr, X_rule]))
    model.fit([X_attr, X_rule], y, epochs=3000, batch_size=5, verbose=1)
    print(model.predict([X_attr, X_rule]))

