from datastruct.graph import GraphEdge

from collections import defaultdict
import numpy as np
from operator import itemgetter
from typing import Dict, Iterable, List, Tuple

import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 10

# TODO currently: model AND dataset as one class; separate in the future

class NeuralModel:
    def __init__(self, edges :List[GraphEdge]):
        self.model_type = 'neural'
        # TODO create rule and edge index
        self.edge_idx = { edge: idx for idx, edge in enumerate(edges) }
        self.rule_idx = { rule: idx for idx, rule in \
                          enumerate(set(edge.rule for edge in edges)) }
        self.ngram_features = self.select_ngram_features(edges)
        self.features = self.ngram_features
        self.X_attr, self.X_rule = self.extract_features_from_edges(edges)
        self.network = self.compile_network()
        self.recompute_edge_prob()

    def fit(self, y :np.ndarray) -> None:
        self.network.fit([self.X_attr, self.X_rule], y, epochs=1000,\
                         batch_size=5, verbose=0)

    def fit_to_sample(self, sample) -> None:
        raise NotImplementedError()

    def recompute_edge_prob(self) -> None:
        self.y_pred = self.network.predict([self.X_attr, self.X_rule])

    def edge_prob(self, edge :GraphEdge) -> float:
        return float(self.y_pred[self.edge_idx[edge]])

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
            for ngram in self.extract_n_grams(edge.source.symstr):
                ngram_freqs[ngram] += 1
        # select most common n-grams
        ngram_features = \
            list(map(itemgetter(0), 
                     sorted(ngram_freqs.items(), 
                            reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
        return ngram_features

    def extract_features_from_edges(self, edges :List[GraphEdge]) -> None:
        attributes, rule_ids = [], []
        for edge in edges:
            attribute_vec = []
            edge_ngrams = set(self.extract_n_grams(edge.source.symstr))
            for ngram in self.ngram_features:
                attribute_vec.append(1 if ngram in edge_ngrams else 0)
            attributes.append(attribute_vec)
            rule_ids.append([self.rule_idx[edge.rule]])
        return np.array(attributes), np.array(rule_ids)

    def compile_network(self):
        num_features, num_rules = len(self.features), len(self.rule_idx)
        input_attr = Input(shape=(num_features,), name='input_attr')
        dense_attr = Dense(100, name='dense_attr')(input_attr)
        input_rule = Input(shape=(1,), name='input_rule')
        rule_emb = Embedding(input_dim=num_rules, output_dim=100,\
                             input_length=1)(input_rule)
        rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
        concat = keras.layers.concatenate([dense_attr, rule_emb_fl])
        dense = Dense(100, activation='softplus', name='dense')(concat)
        output = Dense(1, activation='sigmoid', name='output')(dense)

        model = Model(inputs=[input_attr, input_rule], outputs=[output])
        model.compile(optimizer='sgd', loss='binary_crossentropy')
        return model


