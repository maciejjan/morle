import algorithms.alergia
from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge, FullGraph, Branching
from datastruct.rules import Rule
from models.generic import Model
import shared

from collections import defaultdict
import hfst
import math
import numpy as np
from operator import itemgetter
from typing import Dict, Iterable, List, Tuple
import sys

import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 300
MAX_NEGATIVE_EXAMPLES = 1000000

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

class RootModel:
    def __init__(self, entries :Iterable[LexiconEntry]) -> None:
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()


class AlergiaRootModel:
    def __init__(self, entries :List[LexiconEntry]) -> None:
        word_seqs, tag_seqs = [], []
        for entry in entries:
            word_seqs.append(entry.word)
            tag_seqs.append(entry.tag)

        alpha = shared.config['compile'].getfloat('alergia_alpha')
        freq_threshold = \
            shared.config['compile'].getint('alergia_freq_threshold')
        self.automaton = \
            algorithms.alergia.alergia(word_seqs, alpha=alpha, 
                                       freq_threshold=freq_threshold).to_hfst()
        tag_automaton = \
            algorithms.alergia.prefix_tree_acceptor(tag_seqs).to_hfst()
        tag_automaton.minimize()

        self.automaton.concatenate(tag_automaton)
        self.automaton.remove_epsilons()
        self.automaton.convert(hfst.ImplementationType.HFST_OLW_TYPE)

        self.costs = {}
        for entry in entries:
            self.costs[entry] = self.automaton.lookup(entry.symstr)[0][1]

    def root_cost(self, entry :LexiconEntry) -> float:
        return self.costs[entry]


class EdgeModel:
    def __init__(self, edges :List[GraphEdge], rule_domsizes :Dict[Rule, int])\
                -> None:
        raise NotImplementedError()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def null_cost(self) -> float:
        'Cost of a graph without any edges.'
        raise NotImplementedError()

    def recompute_costs(self) -> None:
        raise NotImplementedError()

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        raise NotImplementedError()


class BernoulliEdgeModel:
    def __init__(self, edges :List[GraphEdge], rule_domsizes :Dict[Rule, int])\
                -> None:
        self.rule_domsizes = rule_domsizes
        self.fit_to_sample(list((edge, 1.0) for edge in edges))

    def edge_cost(self, edge :GraphEdge) -> float:
        return self.rule_cost[edge.rule]

    def null_cost(self) -> float:
        'Cost of a graph without any edges.'
        return self._null_cost

    def recompute_costs(self) -> None:
        # no edge costs are cached, because they are readily obtained
        # from rule costs
        pass

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        rule_freq = { rule: 0.0 for rule in self.rule_domsizes }
        for edge, edge_freq in sample:
            rule_freq[edge.rule] += edge_freq
        self.rule_cost, self._null_cost = {}, 0.0
        for rule in self.rule_domsizes:
            # 0.1 is added to account for a non-uniform prior Beta(1.1, 1.1)
            # (goal: excluding the extreme values 0.0 and 1.0)
            prob = (rule_freq[rule] + 0.1) / (self.rule_domsizes[rule] + 0.2)
            self.rule_cost[rule] = -math.log(prob) + math.log(1-prob)
            self._null_cost -= math.log(1-prob)


class NeuralEdgeModel:
    pass


class FeatureModel:
    def __init__(self, graph :FullGraph) -> None:
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def recompute_costs(self) -> None:
        raise NotImplementedError()

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        raise NotImplementedError()


class NeuralFeatureModel:
    def __init__(self, graph :FullGraph) -> None:
        # TODO edge and root index
        # TODO extract features from edges and roots
        # TODO recompute costs
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def edge_cost(self, edge :GraphEdge) -> float:
        raise NotImplementedError()

    def recompute_costs(self) -> None:
        raise NotImplementedError()

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        raise NotImplementedError()


class ModelSuite:
    def __init__(self, graph :FullGraph, rule_domsizes :Dict[Rule, int])\
                -> None:
        self.graph = graph
        self.root_model = AlergiaRootModel(list(graph.nodes_iter()))
        self.edge_model = BernoulliEdgeModel(list(graph.iter_edges()),
                                             rule_domsizes)
        # TODO feature model -- initially without
        self.reset()

    def cost_of_change(self, edges_to_add :List[GraphEdge],
                       edges_to_delete :List[GraphEdge]) -> float:
        result = 0.0
        for edge in edges_to_add:
            result += self.edge_model.edge_cost(edge)
            result -= self.root_model.root_cost(edge.target)
        for edge in edges_to_delete:
            result -= self.edge_model.edge_cost(edge)
            result += self.root_model.root_cost(edge.target)
        return result

    def apply_change(self, edges_to_add :List[GraphEdge],
                     edges_to_delete :List[GraphEdge]) -> None:
        self._cost += self.cost_of_change(edges_to_add, edges_to_delete)

    def iter_rules(self) -> Iterable[Rule]:
        return self.edge_model.rule_domsizes.keys()
        
    def cost(self) -> float:
        return self._cost

    def reset(self) -> None:
        self._cost = sum(self.root_model.root_cost(entry)\
                        for entry in self.graph.nodes_iter()) +\
                    self.edge_model.null_cost()

    def fit_to_sample(self, sample :List[Tuple[GraphEdge, float]]) -> None:
        self.edge_model.fit_to_sample(sample)
        # TODO fit the feature model

    def fit_to_branching(self, branching :Branching) -> None:
        self.reset()
        self.apply_change(sum(branching.edges_by_rule.values(), []), [])


# TODO this class is deprecated
class NeuralModel(Model):
    def __init__(self, edges :List[GraphEdge]):
        self.model_type = 'neural'
        # create rule and edge index
        self.word_idx = { } # TODO word -> root edge idx
        self.edge_idx = { edge: idx for idx, edge in enumerate(edges, len(self.word_idx)) }
        self.rule_idx = { rule: idx for idx, rule in \
                          enumerate(set(edge.rule for edge in edges)) }
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
            source_seq = edge.source.word + edge.source.tag
            for ngram in self.extract_n_grams(source_seq):
                if ngram in ngram_features_hash:
                    attributes[i, ngram_features_hash[ngram]] = 1
            rule_ids[i, 0] = self.rule_idx[edge.rule]
        print('attributes.nbytes =', attributes.nbytes)
        print('rule_ids.nbytes =', rule_ids.nbytes)
        return attributes, rule_ids

    def sample_negative_examples(self):
        # shuffle the wordlist
        # for word in wordlist:
        #   lookup the word in lexicon .o. rules transducer
        #   for word2:
        #     extract all rules from (word, word2)
        #     for rule:
        #       if rule in ruleset:
        #         add (word, rule) to negative examples
        #         if length(negative examples) >= MAX_NEGATIVE_EXAMPLES:
        #           prepare the weights vector
        #           return
        # if number of examples < MAX_NEGATIVE_EXAMPLES:
        #   resize the array
        # prepare the weights vector
        # return
        raise NotImplementedError()

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


