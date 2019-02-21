#TODO deprecated module

from utils.files import full_path, open_to_write, read_tsv_file, write_line
import shared

from collections import defaultdict

# def load_baseline_data():
#     return sorted(list(set(
#                (min(word_1, word_2), max(word_1, word_2))\
#                 for (word_1, word_2) in read_tsv_file(\
#                     shared.filenames['graph'], types=(str, str)))))

def load_lemmatization():
    result = defaultdict(lambda: list())
    for word, lemma in read_tsv_file('lemmatization.txt'):
        result[word].append(lemma)
    return result

def load_evaluation_data():
    vocab = set(word for (word,) in read_tsv_file(\
                shared.filenames['eval.graph.vocab']))
    # TODO sorting words inside pair: not here but when the files are written!!!
    edges = [(min(word_1, word_2), max(word_1, word_2))\
             for word_1, word_2 in read_tsv_file(
                 shared.filenames['eval.graph'], types=(str, str))]
    edges.sort()
    return vocab, edges

def load_experiment_data():
    reader = read_tsv_file(shared.filenames['sample-wordpair-stats'])
    next(reader)    # skip the header
    edges = [(min(word_1, word_2), max(word_1, word_2), float(prob)) 
             for word_1, word_2, prob in reader if float(prob) > 0.0]
    edges.sort()
    return edges

# def eval_baseline(edges, eval_edges, eval_vocab):
#     i_edges = iter(edges)
#     i_eval_edges = iter(eval_edges)
#     cur_edge = next(i_edges)
#     cur_eval_edge = next(i_eval_edges)
# 
#     tp, fp, fn = 0, 0, 0
# 
#     while True:
#         if cur_edge[:2] == cur_eval_edge[:2]:
#             tp += 1
#             try:
#                 cur_edge = next(i_edges)
#                 cur_eval_edge = next(i_eval_edges)
#             except StopIteration:
#                 break
#         elif cur_edge[:2] < cur_eval_edge[:2]:
#             if cur_edge[0] in eval_vocab and cur_edge[1] in eval_vocab:
#                 fp += 1
#             try:
#                 cur_edge = next(i_edges)
#             except StopIteration:
#                 break
#         else:
#             if cur_eval_edge[0] in eval_vocab and cur_eval_edge[1] in eval_vocab:
#                 fn += 1
#             try:
#                 cur_eval_edge = next(i_eval_edges)
#             except StopIteration:
#                 break
# 
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     fscore = 2 / (1/precision + 1/recall)
# 
#     return precision, recall, fscore

def is_derivation(word_1, word_2, lemmatization):
    pass

def is_inflection(word_1, word_2, lemmatization):
    pass

def is_coinflection(word_1, word_2, lemmatization):
    pass

def evaluate(edges, eval_edges, eval_vocab, prob_bins):
    tp = [0] * len(prob_bins)
    fp = [0] * len(prob_bins)
    fn = [0] * len(prob_bins)

#     i_edges = iter(edges)
#     i_eval_edges = iter(eval_edges)
#     cur_edge = next(i_edges)
#     cur_eval_edge = next(i_eval_edges)
    
    edges_dict = { (word_1, word_2) : prob for word_1, word_2, prob in edges }
    eval_edges_set = set(eval_edges)

    # TP, FP and some FN
    for (word_1, word_2), prob in edges_dict.items():
        if (word_1, word_2) in eval_edges_set:
            for i, pb in enumerate(prob_bins):
                if prob >= pb:
                    tp[i] += 1
                else:
                    fn[i] += 1
        else:
            for i, pb in enumerate(prob_bins):
                if prob >= pb:
                    fp[i] += 1
#                     if prob >= 0.9 and pb >= 0.9:
#                         print(word_1, word_2)
                else:
                    pass

    # the remaining FN
    for (word_1, word_2) in eval_edges_set:
        if (word_1, word_2) not in edges_dict:
            for i, pb in enumerate(prob_bins):
                fn[i] += 1

#     with open_to_write(shared.filenames['eval.report']) as evalfp:
#         while True:
#             if cur_edge[:2] == cur_eval_edge[:2]:
#                 write_line(evalfp, cur_edge + (1.0,))
#                 prob = cur_edge[2]
#                 for i, pb in enumerate(prob_bins):
#                     if prob >= pb:
#                         tp[i] += 1
#                     else:
#                         fn[i] += 1
#                 try:
#                     cur_edge = next(i_edges)
#                     cur_eval_edge = next(i_eval_edges)
#                 except StopIteration:
#                     break
#             elif cur_edge[:2] < cur_eval_edge[:2]:
#                 if cur_edge[0] in eval_vocab and cur_edge[1] in eval_vocab:
#                     write_line(evalfp, cur_edge + (0.0,))
#                     prob = cur_edge[2]
#                     for i, pb in enumerate(prob_bins):
#                         if prob >= pb:
#                             fp[i] += 1
#                 try:
#                     cur_edge = next(i_edges)
#                 except StopIteration:
#                     break
#             else:
#                 if cur_eval_edge[0] in eval_vocab and cur_eval_edge[1] in eval_vocab:
#                     write_line(evalfp, cur_eval_edge + (0.0, 1.0))
#                     for i in range(len(prob_bins)):
#                         fn[i] += 1
#                 try:
#                     cur_eval_edge = next(i_eval_edges)
#                 except StopIteration:
#                     break

    precision, recall, fscore = [], [], []
    for i in range(len(prob_bins)):
        precision.append(tp[i] / (tp[i] + fp[i]))
        recall.append(tp[i] / (tp[i] + fn[i]))
        fscore.append(2 / (1 / precision[-1] + 1 / recall[-1]))

    return precision, recall, fscore

def print_results(precision, recall, fscore, prob_bins):
    for i in range(len(prob_bins)):
        print('{}\t{:2.2f}\t{:2.2f}\t{:2.2f}\t'.format(
            prob_bins[i], precision[i]*100, recall[i]*100, fscore[i]*100))

def run():
    eval_vocab, eval_edges = load_evaluation_data()
#     baseline = load_baseline_data()
    edges = load_experiment_data()
    lemmatization = load_lemmatization()
    prob_bins = [i/1000 for i in range(10)] + [i/100 for i in range(1, 10)] +\
                [i/10 for i in range(1, 10)]

#     precision, recall, fscore = \
#         eval_baseline(baseline, eval_edges, eval_vocab)
#     print('bas\t{:2.2f}\t{:2.2f}\t{:2.2f}\t'.format(
#         precision*100, recall*100, fscore*100))

    print('\n\nOverall:\n')
    precision, recall, fscore = \
        evaluate(edges, eval_edges, eval_vocab, prob_bins)
    print_results(precision, recall, fscore, prob_bins)

#     print('\n\nDerivation:\n')
#     deriv_edges = [(word_1, word_2, prob) for (word_1, word_2, prob) in edges \
#                                if is_derivation(word_1, word_2, lemmatization)]
#     deriv_eval_edges = [(word_1, word_2) for (word_1, word_2) in eval_edges \
#                                if is_derivation(word_1, word_2, lemmatization)]
#     precision, recall, fscore = \
#         evaluate(deriv_edges, deriv_eval_edges, eval_vocab, prob_bins)
#     print_results(precision, recall, fscore, prob_bins)
# 
#     print('\n\nInflection:\n')
#     infl_edges = [(word_1, word_2, prob) for (word_1, word_2, prob) in edges \
#                                if is_inflection(word_1, word_2, lemmatization)]
#     infl_eval_edges = [(word_1, word_2) for (word_1, word_2) in eval_edges \
#                                if is_inflection(word_1, word_2, lemmatization)]
#     precision, recall, fscore = \
#         evaluate(infl_edges, infl_eval_edges, eval_vocab, prob_bins)
#     print_results(precision, recall, fscore, prob_bins)
# 
#     print('\n\nCoinflection:\n')
#     coinfl_edges = [(word_1, word_2, prob) for (word_1, word_2, prob) in edges \
#                              if is_coinflection(word_1, word_2, lemmatization)]
#     coinfl_eval_edges = [(word_1, word_2) for (word_1, word_2) in eval_edges \
#                              if is_coinflection(word_1, word_2, lemmatization)]
#     precision, recall, fscore = \
#         evaluate(coinfl_edges, coinfl_eval_edges, eval_vocab, prob_bins)
#     print_results(precision, recall, fscore, prob_bins)

