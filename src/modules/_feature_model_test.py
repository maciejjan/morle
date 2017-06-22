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

# TODO compare the mean squared error of both prediction methods
# TODO pack into separate classes: two different FeatureModel subclasses
# TODO standard RootModel (ALERGIA) and EdgeModel (Bernoulli)


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


def compute_error(y_true, y_pred):
    vector_dim = y_true.shape[1]
    sq_err = (y_true - y_pred)**2
    return np.sum(sq_err, axis=1) / vector_dim


def compute_baseline(word_idx, edge_idx):
    print('Computing the baseline costs...')
    vector_dim = shared.config['Features'].getint('word_vec_dim')
    edges_by_rule = {}
    for edge in edge_idx:
        if edge.rule not in edges_by_rule:
            edges_by_rule[edge.rule] = []
        edges_by_rule[edge.rule].append(edge)
    # create a matrix of shifts for each word/edge, hashed by rule
    print('* reformatting the data')
    shift_matrix_by_rule = {}
    for rule, edges in edges_by_rule.items():
        shift_matrix_by_rule[rule] =\
            np.array(list(e.target.vec - e.source.vec for e in edges))
    shift_matrix_by_rule['root'] = np.array(list(entry.vec for entry in word_idx))
    # compute the per-rule mean and variance
    print('* fitting the Gaussians')
    means, variances = {}, {}
    for rule, shift_matrix in shift_matrix_by_rule.items():
        n = shift_matrix.shape[0]
        means[rule] = np.sum(shift_matrix, axis=0) / n
        error = shift_matrix - means[rule]
        cov = np.dot(error.T, error) / n
        variances[rule] = np.diag(cov) + np.repeat(0.00001, vector_dim)
    print('* computing the costs and predictions')
    sample_size = len(word_idx) + len(edge_idx)
    costs, pred = np.empty(sample_size), np.empty((sample_size, vector_dim))
    for entry, idx in word_idx.items():
        pred[idx] = means['root']
        costs[idx] = -multivariate_normal.logpdf(entry.vec, means['root'], \
                                                 np.diag(variances['root']))
    for edge, idx in edge_idx.items():
        pred[idx] = edge.source.vec + means[edge.rule]
        costs[idx] = -multivariate_normal.logpdf(\
                          edge.target.vec - edge.source.vec,\
                          means[edge.rule], variances[edge.rule])
    print('* done')
    return costs, pred


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
    baseline_costs, baseline_pred = compute_baseline(word_idx, edge_idx)
    baseline_err = compute_error(y, baseline_pred)
    print('Compiling the model...')
    model = compile_model(X_attr.shape[1], int(np.max(X_rule))+1, y.shape[1])
    model.fit([X_attr, X_rule], y, epochs=5, verbose=1, batch_size=1000)
    y_pred = model.predict([X_attr, X_rule])
    err = compute_error(y, y_pred)
    print('Fitting the error distribution...')
    error_cov = fit_error(y, y_pred)
    print('Computing the edge costs...')
    costs = compute_costs(y, y_pred, error_cov)
    print('Writing output...')
    with open_to_write('_feature_model_test.txt') as outfp:
        for entry, idx in word_idx.items():
            write_line(outfp, ('ROOT', entry, '-', err[idx], costs[idx], 0.0,\
                               baseline_err[idx], baseline_costs[idx], 0.0, 0.0))
#             write_line(outfp, ('ROOT', entry, '-', err[idx], costs[idx], 0.0,\
#                                0.0, 0.0, 0.0, 0.0))
        for edge, idx in edge_idx.items():
            gain = costs[word_idx[edge.target]]-costs[idx]
            baseline_gain = baseline_costs[word_idx[edge.target]]-baseline_costs[idx]
            write_line(outfp, (edge.source, edge.target, edge.rule,
                               err[idx],
                               costs[idx],
                               gain,
                               baseline_err[idx],
                               baseline_costs[idx],
                               baseline_gain,
                               baseline_gain-gain))
#             write_line(outfp, (edge.source, edge.target, edge.rule,
#                                err[idx],
#                                costs[idx],
#                                gain, 0.0, 0.0, 0.0, 0.0))
    print()
    print('Baseline error =', np.sum(baseline_err) / baseline_err.shape[0])
    print('Neural network error =', np.sum(err) / err.shape[0])

