from datastruct.graph import GraphEdge
from datastruct.rules import Rule
from models.neural import NeuralModel

from collections import defaultdict
import numpy as np
from operator import itemgetter
import scipy.sparse
from scipy.special import expit
import tqdm
from typing import Dict, Iterable, List, Tuple

import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model


# MAX_NGRAM_LENGTH = 5
# MAX_NUM_NGRAMS = 10
BATCH_SIZE = 10
COEFF_SD = 0.1

# TODO currently: model AND dataset as one class; separate in the future
# TODO better memory management -- data and predictions in single arrays

class LogitModel(NeuralModel):
    
    # TODO this is going to change

    def __init__(self, edges :List[GraphEdge]):
        self.model_type = 'logit'
        self.num_edges = len(edges)
        self.edges_by_rule = {}
        for edge in edges:
            if edge.rule not in self.edges_by_rule:
                self.edges_by_rule[edge.rule] = []
            self.edges_by_rule[edge.rule].append(edge)
        self.edge_idx = {}
        for rule, r_edges in self.edges_by_rule.items():
            self.edge_idx[rule] = { edge: idx for idx, edge in enumerate(r_edges) }
        self.rule_idx = { rule: idx for idx, rule in \
                          enumerate(self.edge_idx.keys()) }
        self.ngram_features = self.select_ngram_features(edges)
        print(self.ngram_features)
        self.features = self.ngram_features
        self.model_by_rule = self.compile()
        self.X_by_rule = {}
        for rule, r_edges in self.edges_by_rule.items():
            self.X_by_rule[rule] = self.extract_features_from_edges(r_edges)

    def fit_to_sample(self, edges_freq :List[Tuple[GraphEdge, float]]) -> None:
        y_by_rule = {}
        for edge, prob in edges_freq:
            if edge.rule not in y_by_rule:
                y_by_rule[edge.rule] =\
                    np.empty((len(self.edge_idx[edge.rule]),))
            y_by_rule[edge.rule][self.edge_idx[edge.rule][edge]] = prob
        print('Fitting the logit model...')
        progressbar = tqdm.tqdm(total=self.num_edges)
        for rule, model in self.model_by_rule.items():
            model.fit(self.X_by_rule[rule], y_by_rule[rule], epochs=200,\
                      batch_size=32, verbose=0)
            progressbar.update(len(self.edge_idx[rule]))
        progressbar.close()

    def recompute_edge_prob(self) -> None:
        self.y_pred_by_rule = {}
        for rule in self.rule_idx:
            self.y_pred_by_rule[rule] =\
                self.model_by_rule[rule].predict(self.X_by_rule[rule])

    def edge_prob(self, edge :GraphEdge) -> float:
        return float(self.y_pred_by_rule[edge.rule][self.edge_idx[edge.rule][edge]])


    def extract_features_from_edges(self, edges :List[GraphEdge])\
                                   -> Dict[Rule, np.ndarray]:
        attributes = np.zeros((len(edges), len(self.features)))
        ngram_features_hash = {}
        for i, ngram in enumerate(self.ngram_features):
            ngram_features_hash[ngram] = i
        for i, edge in enumerate(edges):
            for ngram in self.extract_n_grams(edge.source.symstr):
                if ngram in ngram_features_hash:
                    row_idx = self.edge_idx[edge.rule][edge]
                    attributes[row_idx, ngram_features_hash[ngram]] = 1
        return attributes

    def compile(self):
        print('Initializing the models...')
        num_features = len(self.features)
        model_by_rule = {}
        for rule in tqdm.tqdm(self.rule_idx):
            input_attr = Input(shape=(num_features,), name='input_attr')
            output = Dense(1, activation='sigmoid', name='output')(input_attr)
            model_by_rule[rule] = Model(inputs=input_attr, outputs=output)
            model_by_rule[rule].compile(optimizer='adam', loss='binary_crossentropy')
        return model_by_rule

