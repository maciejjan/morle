from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge, FullGraph
from models.generic import Model

from collections import defaultdict
import numpy as np
from operator import itemgetter
from typing import Dict, Iterable, List, Tuple
import sys

import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 300

# TODO currently: model AND dataset as one class; separate in the future

# TODO integrate the root probabilities
# TODO routines for running the sampler
# - rules()
# - cost()
# - cost_of_change()
# - apply_change()

# TODO further ideas:
# - take also n-grams of the target word -- useful for e.g. insertion rules
# - take also n-grams around alternation spots

# TODO initialization:
# - root prob = alpha ** word_length - between 0.1 and 0.9
# - edge prob = expit(log(rule freq)) - between 0.1 and 0.9

class NeuralModel(Model):
    def __init__(self, graph :FullGraph):
        self.model_type = 'neural'
        # create rule and edge index
        self.word_idx = { } # TODO word -> root edge idx
        self.edge_idx = { edge: idx for idx, edge in enumerate(edges, len(self.word_idx)) }
        self.rule_idx = { rule: idx for idx, rule in \
                          enumerate(set(edge.rule for edge in edges), 1) }
        self.ngram_features = self.select_ngram_features(edges)
        print(self.ngram_features)
        self.features = self.ngram_features
        self.X_attr, self.X_rule = self.extract_features_from_edges(edges)
        self.network = self.compile()
        self.recompute_edge_prob()

    def fit_to_sample(self, edge_freq :List[Tuple[GraphEdge, float]]) -> None:
        y = np.empty((len(edge_freq),))
        for edge, prob in edge_freq:
            y[self.edge_idx[edge]] = prob
        self.network.fit([self.X_attr, self.X_rule], y, epochs=5,\
                         batch_size=1000, verbose=1)

    # TODO rename -> recompute_edge_costs
    def recompute_edge_prob(self) -> None:
        self.y_pred = self.network.predict([self.X_attr, self.X_rule])

    def edge_prob(self, edge :GraphEdge) -> float:
        return float(self.y_pred[self.edge_idx[edge]])

    def root_prob(self, node :LexiconEntry) -> float:
        raise NotImplementedError()

    def cost() -> float:
        raise NotImplementedError()

    def cost_of_change() -> float:
        raise NotImplementedError()

    def apply_change() -> None:
        raise NotImplementedError()

    def fit_to_branching() -> None:
        # TODO set the cost to the branching cost
        raise NotImplementedError()

    def extract_n_grams(self, word :Iterable[str]) -> Iterable[Iterable[str]]:
        result = []
        max_n = min(MAX_NGRAM_LENGTH, len(word))+1
        result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
        result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
        return result

    def select_ngram_features(self, edges :List[GraphEdge]) -> List[str]:
        # count n-gram frequencies
        ngram_freqs = defaultdict(lambda: 0)
        for edge in edges:
            source_seq = edge.source.word + edge.source.tag
            for ngram in self.extract_n_grams(source_seq):
                ngram_freqs[ngram] += 1
        # select most common n-grams
        ngram_features = \
            list(map(itemgetter(0), 
                     sorted(ngram_freqs.items(), 
                            reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
        return ngram_features

    def extract_features_from_edges(self, edges :List[GraphEdge]) -> None:
        attributes = np.zeros((len(edges), len(self.features)))
        rule_ids = np.zeros((len(edges), 1))
        ngram_features_hash = {}
        for i, ngram in enumerate(self.ngram_features):
            ngram_features_hash[ngram] = i
        print('Memory allocation OK.', file=sys.stderr)
        for i, edge in enumerate(edges):
            for ngram in self.extract_n_grams(edge.source.symstr):
                if ngram in ngram_features_hash:
                    attributes[i, ngram_features_hash[ngram]] = 1
            rule_ids[i, 0] = self.rule_idx[edge.rule]
        print('attributes.nbytes =', attributes.nbytes)
        print('rule_ids.nbytes =', rule_ids.nbytes)
        return attributes, rule_ids

    def compile(self):
        num_features, num_rules = len(self.features), len(self.rule_idx)
        input_attr = Input(shape=(num_features,), name='input_attr')
        dense_attr = Dense(100, activation='softplus', name='dense_attr')(input_attr)
        input_rule = Input(shape=(1,), name='input_rule')
        rule_emb = Embedding(input_dim=num_rules, output_dim=100,\
                             input_length=1)(input_rule)
        rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
        concat = keras.layers.concatenate([dense_attr, rule_emb_fl])
        dense = Dense(100, activation='softplus', name='dense')(concat)
        output = Dense(1, activation='sigmoid', name='output')(dense)

        model = Model(inputs=[input_attr, input_rule], outputs=[output])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model


