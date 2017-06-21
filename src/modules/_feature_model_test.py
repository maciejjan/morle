from datastruct.lexicon import Lexicon, LexiconEntry
from datastruct.graph import FullGraph
from utils.files import open_to_write, write_line
import shared

from collections import defaultdict
import keras
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Model
import numpy as np
from operator import itemgetter
from scipy.stats import multivariate_normal
from typing import Iterable, List


MAX_NGRAM_LENGTH = 5
MAX_NUM_NGRAMS = 300


def normalize_vec(lexicon :Lexicon) -> None:
    for entry in lexicon.entries():
        entry.vec = entry.vec / np.sqrt(np.dot(entry.vec, entry.vec))


def extract_n_grams(word :Iterable[str]) -> Iterable[Iterable[str]]:
    result = []
    max_n = min(MAX_NGRAM_LENGTH, len(word))+1
    result.extend('^'+''.join(word[:i]) for i in range(1, max_n))
    result.extend(''.join(word[-i:])+'$' for i in range(1, max_n))
    return result


def select_ngram_features(entries :Iterable[LexiconEntry]) -> List[str]:
    # count n-gram frequencies
    ngram_freqs = defaultdict(lambda: 0)
    for entry in entries:
        for ngram in extract_n_grams(entry.word + entry.tag):
            ngram_freqs[ngram] += 1
    # select most common n-grams
    ngram_features = \
        list(map(itemgetter(0), 
                 sorted(ngram_freqs.items(), 
                        reverse=True, key=itemgetter(1))))[:MAX_NUM_NGRAMS]
    return ngram_features


def prepare_data(lexicon :Lexicon, graph :FullGraph):
    ngram_features = select_ngram_features(lexicon.entries())
    ngram_features_hash = {}
    for i, ngram in enumerate(ngram_features):
        ngram_features_hash[ngram] = i
    print(ngram_features)
    word_idx = { entry : idx for idx, entry in enumerate(lexicon.entries()) }
    edge_idx = { edge : idx for idx, edge in \
                            enumerate(graph.iter_edges(), len(word_idx)) }
    rule_idx = { rule : idx for idx, rule in \
                            enumerate(set(edge.rule for edge in edge_idx), 1) }
    vector_dim = shared.config['Features'].getint('word_vec_dim')
    sample_size = len(word_idx) + len(edge_idx)
    num_features = len(ngram_features) +\
                   shared.config['Features'].getint('word_vec_dim')
    X_attr = np.zeros((sample_size, num_features))
    X_rule = np.empty(sample_size)
    y = np.empty((sample_size, vector_dim))
    for entry, idx in word_idx.items():
        for ngram in extract_n_grams(entry.word + entry.tag):
            if ngram in ngram_features_hash:
                X_attr[idx, ngram_features_hash[ngram]] = 1
        X_rule[idx] = 0
        y[idx] = entry.vec
    for edge, idx in edge_idx.items():
        for ngram in extract_n_grams(edge.source.word + edge.source.tag):
            if ngram in ngram_features_hash:
                X_attr[idx, ngram_features_hash[ngram]] = 1
        X_attr[idx, len(ngram_features_hash):] = edge.source.vec
        X_rule[idx] = rule_idx[edge.rule]
        y[idx] = edge.target.vec
    return X_attr, X_rule, y, word_idx, edge_idx


def compile_model(num_features, num_rules, vector_dim):
    input_attr = Input(shape=(num_features,), name='input_attr')
    dense_attr = Dense(100, activation='softplus', name='dense_attr')(input_attr)
    input_rule = Input(shape=(1,), name='input_rule')
    rule_emb = Embedding(input_dim=num_rules, output_dim=100,\
                         input_length=1)(input_rule)
    rule_emb_fl = Flatten(name='rule_emb_fl')(rule_emb)
    concat = keras.layers.concatenate([dense_attr, rule_emb_fl])
    output = Dense(vector_dim, activation='linear', name='dense')(concat)

    model = Model(inputs=[input_attr, input_rule], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    return model


def fit_error(y_true, y_pred):
    n = y_true.shape[0]
    error = y_true - y_pred
    return np.dot(error.T, error)/n
    

def compute_costs(y_true, y_pred, error_cov):
    error = y_true - y_pred
    return -multivariate_normal.logpdf(error, np.zeros(y_true.shape[1]), error_cov)


def run():
    print('Loading lexicon...')
    lexicon = Lexicon(shared.filenames['wordlist'])
    normalize_vec(lexicon)
    print('Loading graph...')
    graph = FullGraph(lexicon)
    graph.load_edges_from_file(shared.filenames['graph'])
    print('Preparing data...')
    X_attr, X_rule, y, word_idx, edge_idx = prepare_data(lexicon, graph)
    print('X_attr.shape =', X_attr.shape)
    print('X_rule.shape =', X_rule.shape)
    print('y.shape =', y.shape)
    print('Compiling the model...')
    model = compile_model(X_attr.shape[1], int(np.max(X_rule))+1, y.shape[1])
    model.fit([X_attr, X_rule], y, epochs=5, verbose=1, batch_size=1000)
    y_pred = model.predict([X_attr, X_rule])
    print('Fitting the error distribution...')
    error_cov = fit_error(y, y_pred)
    print('Computing the edge costs...')
    costs = compute_costs(y, y_pred, error_cov)
    print('Writing output...')
    with open_to_write('_feature_model_test.txt') as outfp:
        for entry, idx in word_idx.items():
            write_line(outfp, ('ROOT', entry, '-', costs[idx], 0.0))
        for edge, idx in edge_idx.items():
            write_line(outfp, (edge.source, edge.target, edge.rule, costs[idx],\
                               costs[word_idx[edge.target]]-costs[idx]))

