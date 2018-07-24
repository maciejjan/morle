import algorithms.fst
import algorithms.align
from datastruct.graph import EdgeSet
from datastruct.rules import Rule
import hfst
from utils.files import full_path
import subprocess
from scipy.sparse import csr_matrix

from algorithms.mcmc.statistics import \
    MCMCStatistic, ScalarStatistic, IterationStatistic, EdgeStatistic, \
    RuleStatistic, UnorderedWordPairStatistic
from datastruct.lexicon import LexiconEntry, Lexicon
from datastruct.graph import GraphEdge, Branching, FullGraph
from models.suite import ModelSuite
from utils.files import open_to_write, write_line
import shared

from collections import defaultdict
import logging
import math
import numpy as np
from operator import itemgetter
import random
import sys
import tqdm
from typing import Callable, List, Tuple


class ImpossibleMoveException(Exception):
    pass


# TODO monitor the number of moves from each variant and their acceptance rates!
class MCMCGraphSampler:
    def __init__(self, full_graph :FullGraph, 
                       model :ModelSuite,
                       warmup_iter :int = 1000,
                       sampling_iter :int = 100000,
                       iter_stat_interval :int = 1) -> None:
        self.full_graph = full_graph
        self.lexicon = full_graph.lexicon
        self.edge_set = full_graph.edge_set
        self.rule_set = model.rule_set
        self.model = model
        self.root_cost_cache = np.empty(len(self.lexicon))
        self.edge_cost_cache = np.empty(len(self.edge_set))
        self.warmup_iter = warmup_iter
        self.sampling_iter = sampling_iter
        self.iter_stat_interval = iter_stat_interval
        self.stats = {}               # type: Dict[str, MCMCStatistic]
        self.iter_num = 0

        self.unordered_word_pair_index = {}
        next_id = 0
        for e in self.edge_set:
            key = (min(e.source, e.target), max(e.source, e.target))
            if key not in self.unordered_word_pair_index:
                self.unordered_word_pair_index[key] = next_id
                next_id += 1
    
    def add_stat(self, name: str, stat :MCMCStatistic) -> None:
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def logl(self) -> float:
        return self._logl

    def set_initial_branching(self, branching :Branching) -> None:
        self._logl = \
            float(np.sum(self.root_cost_cache) + self.model.null_cost() +\
                  self.cost_of_change(list(branching.edges_iter()), []))
        logging.getLogger('main').debug('roots cost = {}'\
            .format(np.sum(self.root_cost_cache)))
        logging.getLogger('main').debug('null cost = {}'\
            .format(self.model.null_cost()))
        logging.getLogger('main').debug('initial branching cost = {}'\
            .format(self.cost_of_change(list(branching.edges_iter()), [])))

    def run_sampling(self) -> None:
        self.cache_costs()
        self.branching = self.full_graph.random_branching()
        self.set_initial_branching(self.branching)
        logging.getLogger('main').debug(\
            'initial log-likelihood: {}'.format(self._logl))
        logging.getLogger('main').info('Warming up the sampler...')
        self.reset()
        for i in tqdm.tqdm(range(self.warmup_iter)):
            self.next()
        logging.getLogger('main').debug(\
            'log-likelihood after warmup: {}'.format(self._logl))
        self.reset()
        logging.getLogger('main').info('Sampling...')
        for i in tqdm.tqdm(range(self.sampling_iter)):
            self.next()
        self.update_stats()

    def next(self) -> None:
        # increase the number of iterations
        self.iter_num += 1

        # select an edge randomly
        edge = self.full_graph.random_edge()

        # try the move determined by the selected edge
        try:
            edges_to_add, edges_to_remove, prop_prob_ratio =\
                self.determine_move_proposal(edge)
#             print(len(edges_to_add), len(edges_to_remove))
            acc_prob = self.compute_acc_prob(\
                edges_to_add, edges_to_remove, prop_prob_ratio)
            if acc_prob >= 1 or acc_prob >= random.random():
                self.accept_move(edges_to_add, edges_to_remove)
        # if move impossible -- propose staying in the current graph
        # (the acceptance probability for that is 1, so this move
        # is automatically accepted and nothing needs to be done
        except ImpossibleMoveException:
            pass

        # inform all the statistics that the iteration is completed
        for stat in self.stats.values():
            stat.next_iter()

    # TODO fit to the new Branching class
    # TODO a more reasonable return value?
    def determine_move_proposal(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        if self.branching.has_edge(edge.source, edge.target, edge.rule):
            return self.propose_deleting_edge(edge)
        elif self.branching.has_path(edge.target, edge.source):
            return self.propose_flip(edge)
        elif self.branching.parent(edge.target) is not None:
            return self.propose_swapping_parent(edge)
        else:
            return self.propose_adding_edge(edge)

    def propose_adding_edge(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        return [edge], [], 1

    def propose_deleting_edge(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        return [], [edge], 1

    def propose_flip(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        if random.random() < 0.5:
            return self.propose_flip_1(edge)
        else:
            return self.propose_flip_2(edge)

    def propose_flip_1(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        edges_to_add, edges_to_remove = [edge], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)
        prop_prob_ratio = 1.0
        if node_3 is not None:
            if not self.full_graph.edges_between(node_3, node_1):
                raise ImpossibleMoveException()
            edges_to_add.append(
                random.choice(self.full_graph.edges_between(node_3, node_1)))
            prop_prob_ratio = \
                len(self.full_graph.edges_between(node_3, node_2)) / \
                len(self.full_graph.edges_between(node_3, node_1))
        edges_to_remove.extend(self.branching.edges_between(node_3, node_2))
        edges_to_remove.extend(self.branching.edges_between(node_4, node_1))
        return edges_to_add, edges_to_remove, prop_prob_ratio

    def propose_flip_2(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        edges_to_add, edges_to_remove = [edge], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)
        prop_prob_ratio = 1.0
        if node_3 is not None:
            if not self.full_graph.edges_between(node_3, node_5):
                raise ImpossibleMoveException()
            edges_to_add.append(\
                random.choice(self.full_graph.edges_between(node_3, node_5)))
            prop_prob_ratio = \
                len(self.full_graph.edges_between(node_3, node_2)) / \
                len(self.full_graph.edges_between(node_3, node_5))
        edges_to_remove.extend(self.branching.edges_between(node_2, node_5))
        edges_to_remove.extend(self.branching.edges_between(node_3, node_2))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def nodes_for_flip(self, edge :GraphEdge) -> List[LexiconEntry]:
        node_1, node_2 = edge.source, edge.target
        node_3 = self.branching.parent(node_2)
        node_4 = self.branching.parent(node_1)
        node_5 = node_1
        while self.branching.parent(node_5) != node_2: 
            node_5 = self.branching.parent(node_5)
        return [node_1, node_2, node_3, node_4, node_5]

    def propose_swapping_parent(self, edge :GraphEdge) \
                             -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        edges_to_remove = self.branching.edges_between(
                              self.branching.parent(edge.target),
                              edge.target)
        return [edge], edges_to_remove, 1

    def compute_acc_prob(self, edges_to_add :List[GraphEdge], 
                         edges_to_remove :List[GraphEdge], 
                         prop_prob_ratio :float) -> float:
        cost = self.cost_of_change(edges_to_add, edges_to_remove)
        if cost < math.log(prop_prob_ratio):
            return 1.0
        else: 
            return math.exp(-cost) * prop_prob_ratio

    def cache_costs(self) -> None:
        logging.getLogger('main').info('Computing root costs...')
        self.root_cost_cache = self.model.roots_cost(self.lexicon)
        logging.getLogger('main').info('Computing edge costs...')
        self.edge_cost_cache = self.model.edges_cost(self.edge_set)
        if (np.any(np.isnan(self.root_cost_cache))):
            logging.getLogger('main').warn('NaN in root costs!')
        if (np.any(np.isnan(self.edge_cost_cache))):
            logging.getLogger('main').warn('NaN in edge costs!')
       

    def cost_of_change(self, edges_to_add :List[GraphEdge], 
                       edges_to_remove :List[GraphEdge]) -> float:
        result = 0.0
        for e in edges_to_add:
            result += self.edge_cost_cache[self.edge_set.get_id(e)]
            result -= self.root_cost_cache[self.lexicon.get_id(e.target)]
        for e in edges_to_remove:
            result -= self.edge_cost_cache[self.edge_set.get_id(e)]
            result += self.root_cost_cache[self.lexicon.get_id(e.target)]
        return float(result)

    def accept_move(self, edges_to_add, edges_to_remove):
        self._logl += self.cost_of_change(edges_to_add, edges_to_remove)
        if np.isnan(self._logl):
            logging.getLogger('main').info('adding:')
            for e in edges_to_add:
                print(e.source, e.target, e.rule, \
                      self.edge_cost_cache[self.edge_set.get_id(edge)])
            logging.getLogger('main').info('deleting:')
            for e in edges_to_remove:
                print(e.source, e.target, e.rule, \
                      self.edge_cost_cache[self.edge_set.get_id(edge)])
            raise RuntimeError('NaN log-likelihood at iteration {}'\
                               .format(self.iter_num))
        # remove edges and update stats
        for e in edges_to_remove:
            self.branching.remove_edge(e)
            for stat in self.stats.values():
                stat.edge_removed(e)
        # add edges and update stats
        for e in edges_to_add:
            self.branching.add_edge(e)
            for stat in self.stats.values():
                stat.edge_added(e)
    
    def reset(self):
        self.iter_num = 0
        for stat in self.stats.values():
            stat.reset()

    def update_stats(self):
        for stat in self.stats.values():
            stat.update()

    def print_scalar_stats(self):
        stats, stat_names = [], []
        print()
        print()
        print('SIMULATION STATISTICS')
        print()
        spacing = max([len(stat_name)\
                       for stat_name, stat in self.stats.items() 
                           if isinstance(stat, ScalarStatistic)]) + 2
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, ScalarStatistic):
                print((' ' * (spacing-len(stat_name)))+stat_name, ':', stat.value())
        print()
        print()

    def log_scalar_stats(self):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, ScalarStatistic):
                logging.getLogger('main').info('%s = %f' % (stat_name, stat.value()))

    def save_edge_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, EdgeStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('word_1', 'word_2', 'rule') + tuple(stat_names))
            for idx, edge in enumerate(self.edge_set):
                write_line(fp, 
                           (str(edge.source), str(edge.target), 
                            str(edge.rule)) + tuple([stat.val[idx]\
                                                     for stat in stats]))

    def save_rule_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, RuleStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('rule',) + tuple(stat_names))
            for idx, rule in enumerate(self.rule_set):
                write_line(fp, (str(rule),) +\
                               tuple([stat.val[idx] for stat in stats]))

    def save_wordpair_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, UnorderedWordPairStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('word_1', 'word_2') + tuple(stat_names))
            for key in self.unordered_word_pair_index:
                write_line(fp, key +\
                               tuple([stat.value(key) for stat in stats]))

    def save_iter_stats(self, filename :str) -> None:
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, IterationStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('iter_num',) + tuple(stat_names))
            for iter_num in range(self.iter_stat_interval, 
                                  self.sampling_iter+1, 
                                  self.iter_stat_interval):
                write_line(fp, (str(iter_num),) + \
                               tuple([stat.value(iter_num) for stat in stats]))
            
    def summary(self):
        self.print_scalar_stats()
        self.save_iter_stats(shared.filenames['sample-iter-stats'])
        self.save_edge_stats(shared.filenames['sample-edge-stats'])
        self.save_rule_stats(shared.filenames['sample-rule-stats'])
        self.save_wordpair_stats(shared.filenames['sample-wordpair-stats'])

# TODO constructor arguments should be the same for every type
#      (pass ensured edges through the lexicon parameter?)
# TODO init_lexicon() at creation
class MCMCSemiSupervisedGraphSampler(MCMCGraphSampler):
    def __init__(self, model, lexicon, edges, ensured_conn, warmup_iter, sampl_iter):
        MCMCGraphSampler.__init__(self, model, lexicon, edges, warmup_iter, sampl_iter)
        self.ensured_conn = ensured_conn

    def determine_move_proposal(self, edge):
        edges_to_add, edges_to_remove, prop_prob_ratio =\
            MCMCGraphSampler.determine_move_proposal(self, edge)
        removed_conn = set((e.source, e.target) for e in edges_to_remove) -\
                set((e.source, e.target) for e in edges_to_add)
        if removed_conn & self.ensured_conn:
            raise ImpossibleMoveException()
        else:
            return edges_to_add, edges_to_remove, prop_prob_ratio


class MCMCSupervisedGraphSampler(MCMCGraphSampler):
    def __init__(self, model, lexicon, edges, warmup_iter, sampl_iter):
        logging.getLogger('main').debug('Creating a supervised graph sampler.')
        MCMCGraphSampler.__init__(self, model, lexicon, edges, warmup_iter, sampl_iter)
        self.init_lexicon()

    def init_lexicon(self):
        edges_to_add = []
        for key, edges in self.edges_hash.items():
            edges_to_add.append(random.choice(edges))
        self.accept_move(edges_to_add, [])

    def determine_move_proposal(self, edge):
        if edge in edge.source.edges:
            edge_to_add = random.choice(self.edges_hash[(edge.source, edge.target)])
            if edge_to_add == edge:
                raise ImpossibleMoveException()
            return [edge_to_add], [edge], 1
        else:
            edge_to_remove = self.find_edge_in_lexicon(edge.source, edge.target)
            return [edge], [edge_to_remove], 1

    def run_sampling(self):
        self.reset()
        MCMCGraphSampler.run_sampling(self)


# TODO semi-supervised
class MCMCGraphSamplerFactory:
    def new(*args, **kwargs):
        if shared.config['General'].getboolean('supervised'):
            return MCMCSupervisedGraphSampler(*args, **kwargs)
        else:
            return MCMCGraphSampler(*args, **kwargs)


class MCMCTagSampler(MCMCGraphSampler):

    def __init__(self, full_graph :FullGraph,
                       model :ModelSuite,
                       tagset :List[Tuple[str]],
                       warmup_iter :int = 1000,
                       sampling_iter :int = 100000,
                       iter_stat_interval :int = 1,
                       min_subtree_prob = 1e-100):
#         self.lexicon = full_graph.lexicon
#         self.model = model
        self.tagset = tagset
        logging.getLogger('main').debug('tagset = {}'.format(str(tagset)))
        self.tag_idx = { tag : i for i, tag in enumerate(tagset) }
#         self.warmup_iter = warmup_iter
#         self.sampling_iter = sampling_iter
#         self.iter_stat_interval = iter_stat_interval
        self.min_subtree_prob = min_subtree_prob
#         self.stats = {}
        untagged_edge_set, self.edge_tr_mat = \
            self._compute_untagged_edges_and_transition_mat(full_graph, model)
        untagged_full_graph = FullGraph(full_graph.lexicon, untagged_edge_set)
        super().__init__(untagged_full_graph, model, warmup_iter=warmup_iter,
                         sampling_iter=sampling_iter,
                         iter_stat_interval=iter_stat_interval)
#         self.edge_set = untagged_edge_set
        self._compute_root_prob()
        self._fast_compute_leaf_prob()
        self.init_forward_prob()
        self.init_backward_prob()
        self.write_debug_info()

    def _compute_root_prob(self):
        logging.getLogger('main').info('Computing root probabilities...')
        self.root_prob = \
            np.zeros((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        for w_id, entry in tqdm.tqdm(enumerate(self.lexicon),
                                     total=len(self.lexicon)):
            self.root_prob[w_id,:] = \
                np.exp(-self.model.root_model.root_cost(entry)) * \
                self.model.root_tag_model.predict_tags([entry])

    def _fast_compute_leaf_prob(self):
        logging.getLogger('main').info('Computing leaf probabilities...')
        self.leaf_prob = np.ones((len(self.lexicon), len(self.tagset)))   # ;-)

    def _compute_leaf_prob(self):
        logging.getLogger('main').info('Computing leaf probabilities...')
        self.leaf_prob = np.ones((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        edge_set = EdgeSet(lexicon)

        def _empty_edge_set(edge_set):
            lexicon = edge_set.lexicon
            n = len(edge_set)
            probs = 1-self.model.edges_prob(edge_set)
            for e_id, edge in enumerate(edge_set):
                word = lexicon.get_by_symstr(''.join(edge.source.word))[0]
                w_id = lexicon.get_id(word)
                t_id = self.tag_idx[edge.source.tag]
                self.leaf_prob[w_id,t_id] *= probs[e_id]
            edge_set = EdgeSet(lexicon)
            print(n)
            return edge_set

        lexicon_tr = self.lexicon.to_fst()
        lexicon_tr.concatenate(algorithms.fst.generator(self.tagset))
        rules_tr = self.model.rule_set.to_fst()
        tr = hfst.HfstTransducer(lexicon_tr)
        tr.compose(rules_tr)
        tr.determinize()
        tr.minimize()
        algorithms.fst.save_transducer(tr, 'tr.fsm')
        
        tr_path = full_path('tr.fsm')
        cmd = ['hfst-fst2strings', tr_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, 
                             universal_newlines=True, bufsize=1)
        while True:
            line = p.stdout.readline().strip()
            if line:
                w1, w2 = line.split(':')
                n1 = LexiconEntry(w1)
                n2 = LexiconEntry(w2)
                rules = algorithms.align.extract_all_rules(n1, n2)
                for rule in rules:
                    if rule in rule_set:
                        edge_set.add(GraphEdge(n1, n2, rule))
            else:
                break
            if len(edge_set) > 300000:
                edge_set = _empty_edge_set(edge_set)
        edge_set = _empty_edge_set(edge_set)

    def _compute_untagged_edges_and_transition_mat(self, full_graph, model):
        logging.getLogger('main').info('Computing transition matrices...')

        def _untag_edge(lexicon, edge):
            source = lexicon.get_by_symstr(''.join(edge.source.word))[0]
            target = lexicon.get_by_symstr(''.join(edge.target.word))[0]
            rule = Rule(edge.rule.subst)
            return GraphEdge(source, target, rule)

        edge_prob = model.edges_prob(full_graph.edge_set)
        edge_prob_ratios = edge_prob / (1-edge_prob)
        untagged_edge_set = EdgeSet(full_graph.lexicon)
        T = len(self.tagset)
        edge_ids_by_untagged_edge = []
        for e_id, edge in enumerate(full_graph.edge_set):
            untagged_edge = _untag_edge(full_graph.lexicon, edge)
            if untagged_edge not in untagged_edge_set:
                untagged_edge_set.add(untagged_edge)
                edge_ids_by_untagged_edge.append(list())
            ue_id = untagged_edge_set.get_id(untagged_edge)
            edge_ids_by_untagged_edge[ue_id].append(e_id)
        edge_tr_mat = []
        for ue_id, e_ids in tqdm.tqdm(enumerate(edge_ids_by_untagged_edge), \
                                      total=len(edge_ids_by_untagged_edge)):
            tr_array = np.zeros((T, T))
            for e_id in e_ids:
                edge = full_graph.edge_set[e_id]
                t1_id = self.tag_idx[edge.rule.tag_subst[0]]
                t2_id = self.tag_idx[edge.rule.tag_subst[1]]
                tr_array[t1_id,t2_id] = edge_prob_ratios[e_id]
            if ue_id != len(edge_tr_mat):
                raise Exception('Inconsistent untagged edge IDs!')
            edge_tr_mat.append(csr_matrix(tr_array))
        return untagged_edge_set, edge_tr_mat

    def init_forward_prob(self):
        self.forward_prob = \
            np.empty((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        for w_id in range(len(self.lexicon)):
            self.forward_prob[w_id,:] = self.root_prob[w_id,:]

    def init_backward_prob(self):
        self.backward_prob = \
            np.empty((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        for w_id in range(len(self.lexicon)):
            self.backward_prob[w_id,:] = self.leaf_prob[w_id,:]

    def reset(self):
        self.iter_num = 0
        self.impossible_moves = 0
        self.tag_freq = np.zeros((len(self.lexicon), len(self.tagset)))
        self.last_modified = np.zeros(len(self.lexicon))
        for stat in self.stats.values():
            stat.reset()

    def run_sampling(self) -> None:
        self.branching = self.full_graph.empty_branching()
        self.reset()
        logging.getLogger('main').info('Warming up the sampler...')
        for i in tqdm.tqdm(range(self.warmup_iter)):
            self.next()
        self.reset()
        logging.getLogger('main').info('Sampling...')
        for i in tqdm.tqdm(range(self.sampling_iter)):
            self.next()
        self.finalize()

    def compute_acc_prob(self, edges_to_add, edges_to_remove, prop_prob_ratio):
        if len(edges_to_add) == 1 and len(edges_to_remove) == 0:
            tgt_id = self.lexicon.get_id(edges_to_add[0].target)
            prob = np.sum(self.root_prob[tgt_id,:]*self.backward_prob[tgt_id,:])
            if prob == 0:
                return 0
            return self.compute_acc_prob_for_subtree(\
                       edges_to_add, edges_to_remove) / prob * prop_prob_ratio
        elif len(edges_to_add) == 0 and len(edges_to_remove) == 1:
            tgt_id = self.lexicon.get_id(edges_to_remove[0].target)
            prob = np.sum(self.root_prob[tgt_id,:]*self.backward_prob[tgt_id,:])
            return self.compute_acc_prob_for_subtree(\
                       edges_to_add, edges_to_remove) * prob * prop_prob_ratio
        else:
            edges_to_change_by_root = {}
            for edge in edges_to_add:
                root = self.branching.root(edge.source)
                if root not in edges_to_change_by_root:
                    edges_to_change_by_root[root] = (list(), list())
                edges_to_change_by_root[root][0].append(edge)
            for edge in edges_to_remove:
                root = self.branching.root(edge.source)
                if root not in edges_to_change_by_root:
                    edges_to_change_by_root[root] = (list(), list())
                edges_to_change_by_root[root][1].append(edge)
            prob = 1.0
            for root, (edges_to_add, edges_to_remove) in \
                    edges_to_change_by_root.items():
                logging.getLogger('main').debug(\
                    'edges_to_change_by_root[{}] = {} ;;; {}'\
                    .format(str(root),
                            '; '.join(str(edge) for edge in edges_to_add),
                            '; '.join(str(edge) for edge in edges_to_remove)))
                prob *= self.compute_acc_prob_for_subtree(\
                            edges_to_add, edges_to_remove)
            return prob * prop_prob_ratio

    def compute_acc_prob_for_subtree(self, edges_to_add, edges_to_remove):

        def _build_modified_subtree(edges_to_add, edges_to_remove):
            '''Return a subset of self.branching with `edges_to_add` added 
               and `edges_to_remove` removed. The returned subset contains
               all nodes, for which the backward probability changes,
               together with their children. It is a tree rooted in the common
               ancestor of source nodes of changed edges, plus target nodes
               of edges that would introduce a cycle (in case of the "flip"
               move.'''

            def _common_ancestor(node_1, node_2, depth_1=None, depth_2=None):
                if node_1 == node_2:
                    return node_1
                if depth_1 is None:
                    depth_1 = self.branching.depth(node_1)
                if depth_2 is None:
                    depth_2 = self.branching.depth(node_2)
                if max(depth_1, depth_2) <= 0:
                    return None
                if depth_1 < depth_2:
                    return _common_ancestor(node_1, self.branching.parent(node_2),\
                                            depth_1, depth_2-1)
                elif depth_1 > depth_2:
                    return _common_ancestor(self.branching.parent(node_1), node_2,\
                                            depth_1-1, depth_2)
                else:
                    return _common_ancestor(self.branching.parent(node_1), \
                                            self.branching.parent(node_2), \
                                            depth_1-1, depth_2-1)

            def _add_outgoing_edges(branching, node):
                for oe in self.branching.outgoing_edges(node):
                    if not branching.has_edge(oe.source, oe.target, oe.rule):
                        branching.add_edge(oe)

            edges_to_change = edges_to_add + edges_to_remove
            subtree_root = edges_to_change[0].source
            for edge in edges_to_change[1:]:
                subtree_root = _common_ancestor(subtree_root, edge.source)
            root_of_subtree_root = self.branching.root(subtree_root)
            for edge in edges_to_change:
                if self.branching.root(edge.target) == root_of_subtree_root:
                    subtree_root = _common_ancestor(subtree_root, edge.target)
            modified_subtree = Branching()
            modified_nodes = set()
            # create a copy of a fragment of the branching
            # containing paths from subtree_root to each source of a changed
            # edge
            for src in [edge.source for edge in edges_to_change]:
                node = src
                modified_nodes.add(node)
                _add_outgoing_edges(modified_subtree, node)
                while node != subtree_root:
                    ie = self.branching.ingoing_edges(node)[0]
                    modified_subtree.add_edge(ie)
                    if ie.source not in modified_nodes:
                        _add_outgoing_edges(modified_subtree, ie.source)
                        modified_nodes.add(ie.source)
                    node = ie.source
            # perform the changes on our copy
            for edge in edges_to_add:
                modified_subtree.add_edge(edge)
                if edge.target not in modified_nodes:
                    _add_outgoing_edges(modified_subtree, edge.target)
                    modified_nodes.add(edge.target)
            for edge in edges_to_remove:
                modified_subtree.remove_edge(edge)
            return modified_subtree, modified_nodes

        def _list_nodes_breadth_first(branching, nodes):
            # list tree nodes in the order of increasing depth
            queue = [branching.root(list(nodes)[0])]
            result = []
#             print([str(n) for n in queue])
            while queue:
                node = queue.pop(0)
                result.append(node)
                for successor in branching.successors(node):
                    if successor in nodes:
                        queue.append(successor)
#                 print([str(n) for n in queue])
            return result

        def _recompute_backward_prob(branching, nodes):
            # recompute the backward probabilities for the given nodes
            # (in reverse order) according to the given branching
            # TODO refactor -- code duplication with
            #      MCMCTagSampler.recompute_backward_prob_for_node()
            new_backward_prob = np.empty((len(nodes), len(self.tagset)),
                                         dtype=np.float64)
            for i in range(len(nodes)-1, -1, -1):
                node = nodes[i]
                w_id = self.lexicon.get_id(node)
                new_backward_prob[i,:] = self.leaf_prob[w_id,:]
                for edge in branching.outgoing_edges(node):
                    e_id = self.full_graph.edge_set.get_id(edge)
                    b = new_backward_prob[nodes.index(edge.target),:] \
                        if edge.target in nodes \
                        else self.backward_prob[self.lexicon.get_id(edge.target),:]
                    new_backward_prob[i,:] *= self.edge_tr_mat[e_id].dot(b)
            return new_backward_prob

        def _recompute_root_forward_prob(branching, root, edges_to_remove):
            r_id = self.lexicon.get_id(root)
            if self.branching.parent(root) is not None and \
                   self.branching.ingoing_edges(root)[0] \
                        not in edges_to_remove:
                # the parent of our subtree root is outside of the changed
                # subtree -> forward probability did not change
                return np.copy(self.forward_prob[r_id,:])
            else:
                # the former parent of our root is inside the changed subtree
                # and its deriving edge was removed, or our subtree root
                # was previously a true tree root; either way, now it is
                # a true root
                return np.copy(self.root_prob[r_id,:])

        modified_subtree, modified_nodes = \
            _build_modified_subtree(edges_to_add, edges_to_remove)
        modified_nodes = \
            _list_nodes_breadth_first(modified_subtree, modified_nodes)
        new_backward_prob = \
            _recompute_backward_prob(modified_subtree, modified_nodes)
        new_root_forward_prob = \
           _recompute_root_forward_prob(modified_subtree, modified_nodes[0],
                                        edges_to_remove)
        r_id = self.lexicon.get_id(modified_nodes[0])

        old_prob = np.sum(self.forward_prob[r_id,:] * \
                          self.backward_prob[r_id,:])
        new_prob = np.sum(new_root_forward_prob * new_backward_prob[0,:])
        acc_prob = 0 \
                   if new_prob < self.min_subtree_prob \
                   else new_prob / old_prob
        return acc_prob

    def recompute_forward_prob_for_node(self, node):
        w_id = self.lexicon.get_id(node)
        old_value = np.copy(self.forward_prob[w_id,:])
        if self.branching.parent(node) is None:
            result = np.copy(self.root_prob[w_id,:])
        else:
            parent = self.branching.parent(node)
            par_id = self.lexicon.get_id(parent)
            result = np.copy(self.forward_prob[par_id,:])
            for edge in self.branching.outgoing_edges(parent):
                if edge.target != node:
                    e_id = self.full_graph.edge_set.get_id(edge)
                    w2_id = self.lexicon.get_id(edge.target)
                    result *= self.edge_tr_mat[e_id].dot(self.backward_prob[w2_id,:])
            e_id = self.full_graph.edge_set.get_id(\
                       self.branching.ingoing_edges(node)[0])
            result = np.dot(result, self.edge_tr_mat[e_id].toarray())
        self.forward_prob[w_id,:] = result
        if not np.any(result > 0):
            root = self.branching.root(node)
            parent = self.branching.parent(node)
            print()
            print('root =', root)
            print('subtree size =', self.branching.subtree_size(root))
            print('subtree height =', self.branching.height(root))
            print('root prob.=', self.root_prob[self.lexicon.get_id(root),:])
            print('parent forward prob.=', self.forward_prob[self.lexicon.get_id(parent),:])
            print('Zero forward prob. for: {} (old value = {})'\
                  .format(self.lexicon[w_id], old_value))
            raise Exception()

    def recompute_forward_prob_for_subtree(self, root):
        self.recompute_forward_prob_for_node(root)
        for node in self.branching.successors(root):
            self.recompute_forward_prob_for_subtree(node)

    def recompute_backward_prob_for_node(self, node):
        w_id = self.lexicon.get_id(node)
        result = np.copy(self.leaf_prob[w_id,:])
        for edge in self.branching.outgoing_edges(node):
            e_id = self.full_graph.edge_set.get_id(edge)
            w2_id = self.lexicon.get_id(edge.target)
            result *= self.edge_tr_mat[e_id].dot(self.backward_prob[w2_id,:])
        self.backward_prob[w_id,:] = result
        if not np.any(result > 0):
            raise Exception('Zero backward prob. for: {}'.format(self.lexicon[w_id]))

    def recompute_backward_prob_for_subtree(self, root):
        for node in self.branching.successors(root):
            self.recompute_backward_prob_for_subtree(node)
        self.recompute_backward_prob_for_node(root)

    def check_probs_for_subtree(self, root, value=None):
        w_id = self.lexicon.get_id(root)
        if value is None:
            value = np.sum(self.forward_prob[w_id,:] * self.backward_prob[w_id,:])
        if not np.isclose(value, np.sum(self.forward_prob[w_id,:] * \
                                        self.backward_prob[w_id,:])):
            return False
        for child in self.branching.successors(root):
            if not self.check_probs_for_subtree(child, value):
                return False
        return True

    def update_tag_freq_for_node(self, node):
        w_id = self.lexicon.get_id(node)
        cur_tag_freq = self.forward_prob[w_id,:] * self.backward_prob[w_id,:]
        cur_tag_freq_sum = np.sum(cur_tag_freq)
        if cur_tag_freq_sum > 0:
            cur_tag_freq /= np.sum(cur_tag_freq)
        self.tag_freq[w_id,:] = \
            (self.tag_freq[w_id,:] * self.last_modified[w_id] + \
             cur_tag_freq * (self.iter_num - self.last_modified[w_id])) /\
            self.iter_num
        self.last_modified[w_id] = self.iter_num

    def update_tag_freq_for_subtree(self, root):
        self.update_tag_freq_for_node(root)
        for node in self.branching.successors(root):
            self.update_tag_freq_for_subtree(node)

    def accept_move(self, edges_to_add, edges_to_remove):
        # remove edges and update stats
        roots_changed = set()
        for e in edges_to_remove:
            self.branching.remove_edge(e)
            roots_changed.add(self.branching.root(e.source))
            roots_changed.add(e.target)
            for stat in self.stats.values():
                stat.edge_removed(e)
        # add edges and update stats
        for e in edges_to_add:
            self.branching.add_edge(e)
            roots_changed.add(self.branching.root(e.source))
            for stat in self.stats.values():
                stat.edge_added(e)
        roots_changed = { root for root in roots_changed \
                          if self.branching.parent(root) is None }
        for root in roots_changed:
            self.update_tag_freq_for_subtree(root)
        for root in roots_changed:
            self.recompute_backward_prob_for_subtree(root)
        for root in roots_changed:
            self.recompute_forward_prob_for_subtree(root)

    def finalize(self):
        for node in self.lexicon:
            self.update_tag_freq_for_node(node)
        for stat in self.stats.values():
            stat.update()

    def write_root_prob(self, filename):
        with open_to_write(filename) as fp:
            for w_id, entry in enumerate(self.lexicon):
                tag_probs = [''.join(tag)+':'+str(self.root_prob[w_id,t_id]) \
                             for t_id, tag in enumerate(self.tagset)]
                write_line(fp, (str(entry), ' '.join(tag_probs)))

    def write_leaf_prob(self, filename):
        with open_to_write(filename) as fp:
            for w_id, entry in enumerate(self.lexicon):
                tag_probs = [''.join(tag)+':'+str(self.leaf_prob[w_id,t_id]) \
                             for t_id, tag in enumerate(self.tagset)]
                write_line(fp, (str(entry), ' '.join(tag_probs)))

    def write_edge_tr_mat(self, filename):
        with open_to_write(filename) as fp:
            for e_id, edge in enumerate(self.full_graph.edge_set):
                tag_probs = []
                edge_tr_mat = self.edge_tr_mat[e_id]
                for (t1_id, t2_id), val in edge_tr_mat.todok().items():
                    tag_1 = self.tagset[t1_id]
                    tag_2 = self.tagset[t2_id]
                    tag_probs.append((''.join(tag_1), ''.join(tag_2), str(val)))
                write_line(fp, (str(edge), ' '.join([t1+':'+t2+':'+prob \
                                                     for t1, t2, prob in tag_probs])))

    def write_debug_info(self):
        self.write_root_prob('sampler-root-prob.txt')
#         self.write_leaf_prob('sampler-leaf-prob.txt')
        self.write_edge_tr_mat('sampler-edge-tr-mat.txt')

