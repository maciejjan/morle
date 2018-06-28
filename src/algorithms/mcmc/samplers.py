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
#         self.reset()

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
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.full_graph.has_edge(node_3, node_1):
            raise ImpossibleMoveException()

        edge_3_1 = random.choice(self.full_graph.edges_between(node_3, node_1))
        edge_3_2 = self.branching.edges_between(node_3, node_2)[0] \
                   if self.branching.has_edge(node_3, node_2) else None
        edge_4_1 = self.branching.edges_between(node_4, node_1)[0] \
                   if self.branching.has_edge(node_4, node_1) else None

        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        if edge_4_1 is not None:
            edges_to_remove.append(edge_4_1)
        else: raise Exception('!')
        edges_to_add.append(edge_3_1)
        prop_prob_ratio = (1/len(self.full_graph.edges_between(node_3, node_1))) /\
                          (1/len(self.full_graph.edges_between(node_3, node_2)))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def propose_flip_2(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.full_graph.has_edge(node_3, node_5):
            raise ImpossibleMoveException()

        edge_2_5 = self.branching.edges_between(node_2, node_5)[0] \
                   if self.branching.has_edge(node_2, node_5) else None
        edge_3_2 = self.branching.edges_between(node_3, node_2)[0] \
                   if self.branching.has_edge(node_3, node_2) else None
        edge_3_5 = random.choice(self.full_graph.edges_between(node_3, node_5))

        if edge_2_5 is not None:
            edges_to_remove.append(edge_2_5)
        elif node_2 != node_5: raise Exception('!')     # TODO ???
        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        edges_to_add.append(edge_3_5)
        prop_prob_ratio = (1/len(self.full_graph.edges_between(node_3, node_5))) /\
                          (1/len(self.full_graph.edges_between(node_3, node_2)))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def nodes_for_flip(self, edge :GraphEdge) -> List[LexiconEntry]:
        node_1, node_2 = edge.source, edge.target
        node_3 = self.branching.parent(node_2)
        node_4 = self.branching.parent(node_1)
        node_5 = node_4
        if node_5 != node_2:
            while self.branching.parent(node_5) != node_2: 
                node_5 = self.branching.parent(node_5)
        return [node_1, node_2, node_3, node_4, node_5]

    def find_edge_in_full_graph(self, source, target):
        edges = [e for e in source.edges if e.target == target] 
        return edges[0] if edges else None

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
#         try:
#             return math.exp(\
#                     -self.cost_of_change(edges_to_add, edges_to_remove)) *\
#                    prop_prob_ratio
#         except OverflowError as e:
#             logging.getLogger('main').debug('OverflowError')
#             cost = -self.cost_of_change(edges_to_add, edges_to_remove)
#             if edges_to_add:
#                 logging.getLogger('main').debug('adding:')
#                 for edge in edges_to_add:
#                     logging.getLogger('main').debug(
#                         '{} {} {} {}'.format(edge.source, edge.target,
#                                              edge.rule, self.model.edge_cost(edge)))
#             if edges_to_remove:
#                 logging.getLogger('main').debug('deleting:')
#                 for edge in edges_to_remove:
#                     logging.getLogger('main').debug(
#                         '{} {} {} {}'.format(edge.source, edge.target,
#                                              edge.rule, -self.model.edge_cost(edge)))
#             logging.getLogger('main').debug('total cost: {}'.format(cost))
#             return 1.0

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
#             self.model.apply_change([], [e])
            for stat in self.stats.values():
                stat.edge_removed(e)
        # add edges and update stats
        for e in edges_to_add:
            self.branching.add_edge(e)
#             self.model.apply_change([e], [])
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


class MCMCTagSamplerRootsOnly:

    def __init__(self, lexicon :Lexicon, 
                       model :ModelSuite,
                       tagset :List[Tuple[str]],
                       warmup_iter :int = 1000,
                       sampling_iter :int = 100000,
                       iter_stat_interval :int = 1) -> None:
        self.lexicon = lexicon
        self.model = model
        self.warmup_iter = warmup_iter
        self.sampling_iter = sampling_iter
        self.iter_stat_interval = iter_stat_interval
        self.stats = {}               # type: Dict[str, MCMCStatistic]
        # fields related to the tags
        self.tagset = tagset
#         self.tag_idx = { tag : i for i, tag in enumerate(tagset) }
        self.current_tag = \
            np.random.randint(len(self.tagset), size=len(self.lexicon))
        self.root_cost_cache = np.empty((len(self.lexicon), len(self.tagset)))
        print('Computing root costs...')
        for w_id, entry in tqdm.tqdm(enumerate(lexicon), total=len(lexicon)):
            for t_id, tag in enumerate(tagset):
                self.root_cost_cache[w_id,t_id] = \
                    self.model.root_cost(LexiconEntry(''.join(entry.word) + \
                                                      ''.join(tag)))
        self.reset()

    def run_sampling(self):
        logging.getLogger('main').info('Warming up the sampler...')
        self.reset()
        for i in tqdm.tqdm(range(self.warmup_iter)):
            self.next()
        self.reset()
        logging.getLogger('main').info('Sampling...')
        for i in tqdm.tqdm(range(self.sampling_iter)):
            self.next()
        self.finalize()

    def reset(self):
        self.iter_num = 0
        self.tag_freq = np.zeros((len(self.lexicon), len(self.tagset)))
        self.last_modified = np.zeros(len(self.lexicon))

    def next(self):
        # increase the number of iterations
        self.iter_num += 1

        # select a root and a new tag randomly
        w_id = np.random.randint(len(self.lexicon))
        tag_id = np.random.randint(len(self.tagset))

        # compute the cost of retagging
#         cost = self.model.root_cost(new_root) - self.model.root_cost(old_root)
        cost = self.root_cost_cache[w_id,tag_id] - \
               self.root_cost_cache[w_id,self.current_tag[w_id]]
        acc_prob = 1.0 if cost < 0 else math.exp(-cost)
        if random.random() < acc_prob:
            # accept move
            last_tag_one_hot = np.zeros(len(self.tagset))
            last_tag_one_hot[self.current_tag[w_id]] = 1
            self.tag_freq[w_id,:] = \
                self.tag_freq[w_id,:] * \
                    (self.last_modified[w_id] / self.iter_num) + \
                last_tag_one_hot * \
                    ((self.iter_num-self.last_modified[w_id]) / self.iter_num)
            self.current_tag[w_id] = tag_id
            self.last_modified[w_id] = self.iter_num

    def finalize(self):
        self.last_tag = np.zeros((len(self.lexicon), len(self.tagset)))
        for i in range(len(self.lexicon)):
            self.last_tag[i, self.current_tag[i]] = 1
        T = len(self.tagset)
        self.tag_freq = \
            self.tag_freq * \
            np.tile(self.last_modified / self.iter_num, (T, 1)).T + \
            self.last_tag * \
                np.tile((self.iter_num-self.last_modified) / self.iter_num,
                        (T, 1)).T


class MCMCTagSamplerMove:
    
    def __init__(self, sampler :'MCMCTagSampler') -> None:
        self.sampler = sampler
        self.nodes_to_retag = {}
        self.edges_to_add = set()
        self.edges_to_remove = set()
        self.roots_to_add = set()
        self.roots_to_remove = set()

    def __bool__(self):
        return bool(self.nodes_to_retag) or bool(self.edges_to_add) or \
               bool(self.edges_to_remove)

    def cost(self) -> float:
        result = 0.0
        for edge in self.edges_to_add:
            e_id = self.sampler.edge_set.get_id(edge)
            result += self.sampler.edge_cost_cache[e_id]
        for edge in self.edges_to_remove:
            e_id = self.sampler.edge_set.get_id(edge)
            result -= self.sampler.edge_cost_cache[e_id]
        for (node, tag) in self.roots_to_add:
            w_id = self.sampler.lexicon.get_id(node)
            t_id = self.sampler.tag_idx[tag]
            result += self.sampler.root_cost_cache[w_id,t_id]
        for (node, tag) in self.roots_to_remove:
            w_id = self.sampler.lexicon.get_id(node)
            t_id = self.sampler.tag_idx[tag]
            result -= self.sampler.root_cost_cache[w_id,t_id]
        return result

    def add_edge(self, edge :GraphEdge) -> None:
        self._add_edge(edge)
        self._remove_root(edge.target,
                          self.sampler.get_current_tag(edge.target))
        self.retag_node(edge.source, edge.rule.tag_subst[0])
        self.nodes_to_retag[edge.target] = edge.rule.tag_subst[1]
        for outgoing_edge in self.sampler.branching.outgoing_edges(edge.target):
            self.retag_edge(outgoing_edge, edge.rule.tag_subst[1],
                            outgoing_edge.rule.tag_subst[1])

    def remove_edge(self, edge :GraphEdge) -> None:
        self._remove_edge(edge)
        self._add_root(edge.target, self.sampler.get_current_tag(edge.target))

    def retag_node(self, node :LexiconEntry, tag :Tuple[str]) -> None:
        '''Changes the tag of the node to the desired tag. Triggers retagging
           the ingoing edge and the outgoing edges of the node.'''
        if tag != self.sampler.get_current_tag(node):
            self.nodes_to_retag[node] = tag
            if self.sampler.branching.parent(node) is None:
                # retag as root
                self._remove_root(node, self.sampler.get_current_tag(node))
                self._add_root(node, tag)
            else:
                # retag the ingoing edge
                ingoing_edge = list(self.sampler.branching.ingoing_edges(node))[0]
                self.retag_edge(\
                    ingoing_edge, ingoing_edge.rule.tag_subst[0], tag)
            # retag the outgoing edges
            for edge in self.sampler.branching.outgoing_edges(node):
                self.retag_edge(edge, tag, edge.rule.tag_subst[1])

    def retag_edge(self,
                   edge :GraphEdge,
                   new_source_tag :Tuple[str],
                   new_target_tag :Tuple[str]) -> None:
        self._remove_edge(edge)
        new_edges = \
            [e for e in self.sampler.full_graph.edges_between(\
                            edge.source, edge.target) \
                        if e.rule.tag_subst[0] == new_source_tag \
                                and e.rule.tag_subst[1] == new_target_tag]
        if not new_edges:
            self._add_root(edge.target, self.sampler.get_current_tag(edge.target))
        else:
            self._add_edge(random.choice(new_edges))

    def _add_edge(self, edge :GraphEdge) -> None:
        if edge in self.edges_to_remove:
            self.edges_to_remove.remove(edge)
        else:
            self.edges_to_add.add(edge)
    
    def _remove_edge(self, edge :GraphEdge) -> None:
        if edge in self.edges_to_add:
            self.edges_to_add.remove(edge)
        else:
            self.edges_to_remove.add(edge)

    def _add_root(self, node :LexiconEntry, tag :Tuple[str]) -> None:
        if (node, tag) in self.roots_to_remove:
            self.roots_to_remove.remove((node, tag))
        else:
            self.roots_to_add.add((node, tag))

    def _remove_root(self, node :LexiconEntry, tag :Tuple[str]) -> None:
        if (node, tag) in self.roots_to_add:
            self.roots_to_add.remove((node, tag))
        else:
            self.roots_to_remove.add((node, tag))


class MCMCTagSampler:

    def __init__(self, full_graph :FullGraph,
                       model :ModelSuite,
                       tagset :List[Tuple[str]],
                       warmup_iter :int = 1000,
                       sampling_iter :int = 100000,
                       iter_stat_interval :int = 1,
                       temperature_fun :Callable = None) -> None:
        self.full_graph = full_graph
        self.lexicon = full_graph.lexicon
        self.edge_set = full_graph.edge_set
        self.model = model
        self.warmup_iter = warmup_iter
        self.sampling_iter = sampling_iter
        self.iter_stat_interval = iter_stat_interval
        self.temperature_fun = temperature_fun
        self.stats = {}               # type: Dict[str, MCMCStatistic]
        self.branching = self.full_graph.empty_branching()
        # fields related to the tags
        self.tagset = tagset
        self.tag_idx = { tag : i for i, tag in enumerate(tagset) }
        self.current_tag = \
            np.random.randint(len(self.tagset), size=len(self.lexicon))
        self.root_cost_cache = np.empty((len(self.lexicon), len(self.tagset)))
        # TODO move to a separate method: cache_costs()
        print('Computing root costs...')
        for w_id, entry in tqdm.tqdm(enumerate(self.lexicon), total=len(self.lexicon)):
            for t_id, tag in enumerate(tagset):
                self.root_cost_cache[w_id,t_id] = \
                    self.model.root_cost(LexiconEntry(''.join(entry.word) + \
                                                      ''.join(tag)))
        print('Computing edge costs...')
        self.edge_cost_cache = self.model.edges_cost(self.edge_set)
        self.reset()

    def get_current_tag(self, node :LexiconEntry) -> Tuple[str]:
        return self.tagset[self.current_tag[self.lexicon.get_id(node)]]

    def add_stat(self, name: str, stat :MCMCStatistic) -> None:
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def run_sampling(self):
        logging.getLogger('main').info('Warming up the sampler...')
        self.reset()
        for i in tqdm.tqdm(range(self.warmup_iter)):
            self.next()
        self.reset()
        logging.getLogger('main').info('Sampling...')
        for i in tqdm.tqdm(range(self.sampling_iter)):
            self.next()
        print('Impossible moves: {} ({:.2} %)'\
                  .format(self.impossible_moves,
                          self.impossible_moves / self.sampling_iter * 100))
        print()
        print('accepted moves balance:')
        for key in sorted(self.accepted_moves):
            print(key, self.accepted_moves[key])
        print()
        print('rejected moves balance:')
        for key in sorted(self.rejected_moves):
            print(key, self.rejected_moves[key])
        self.finalize()

    def reset(self):
        '''Reset all statistics.'''
        self.iter_num = 0
        self.impossible_moves = 0
        self.accepted_moves = defaultdict(lambda: 0)
        self.rejected_moves = defaultdict(lambda: 0)
        self.tag_freq = np.zeros((len(self.lexicon), len(self.tagset)))
        self.last_modified = np.zeros(len(self.lexicon))
        for stat in self.stats.values():
            stat.reset()

    def next(self):
        self.iter_num += 1
        try:
            move = self.choose_and_change_a_node() \
                   if random.random() < 0.5 \
                   else self.choose_and_change_an_edge()
            cost = move.cost()
            temperature = self.temperature_fun(self.iter_num) \
                          if self.temperature_fun is not None \
                          else 1.0
            acc_prob = 1.0 if cost < 0 else math.exp(-cost/temperature)
            balance = len(move.edges_to_add) - len(move.edges_to_remove)
            if move and random.random() < acc_prob:
                self.accept_move(move)
                self.accepted_moves[balance] += 1
            else:
                self.rejected_moves[balance] += 1

        except ImpossibleMoveException:
            self.impossible_moves += 1
#             pass


    def accept_move(self, move :MCMCTagSamplerMove) -> None:
        # TODO stat.move_accepted()
        #      (bug: acceptance rate = 0 while sampling without edges)
        for node, tag in move.nodes_to_retag.items():
            w_id = self.lexicon.get_id(node)
            tag_id = self.tag_idx[tag]
            last_tag_one_hot = np.zeros(len(self.tagset))
            last_tag_one_hot[self.current_tag[w_id]] = 1
            self.tag_freq[w_id,:] = \
                self.tag_freq[w_id,:] * \
                    (self.last_modified[w_id] / self.iter_num) + \
                last_tag_one_hot * \
                    ((self.iter_num-self.last_modified[w_id]) / self.iter_num)
            self.current_tag[w_id] = tag_id
            self.last_modified[w_id] = self.iter_num
        for edge in move.edges_to_add:
#             edge = self.edge_set[e_id]
#             print('Adding edge: {} {} {}'.format(edge.source, edge.target, str(edge.rule)))
            self.branching.add_edge(edge)
            for stat in self.stats.values():
                stat.edge_added(edge)
        for edge in move.edges_to_remove:
#             edge = self.edge_set[e_id]
#             print('Deleting edge: {} {} {}'.format(edge.source, edge.target, str(edge.rule)))
            self.branching.remove_edge(edge)
            for stat in self.stats.values():
                stat.edge_removed(edge)

    def choose_and_change_a_node(self) -> MCMCTagSamplerMove:
        # select a root and a new tag randomly
        w_id = np.random.randint(len(self.lexicon))
        tag_id = np.random.randint(len(self.tagset))
        return self.propose_retagging_node(self.lexicon[w_id], self.tagset[tag_id])

    def choose_and_change_an_edge(self) -> MCMCTagSamplerMove:
        edge = self.full_graph.random_edge()

        if self.branching.has_edge(edge.source, edge.target, edge.rule):
            return self.propose_deleting_edge(edge)
        elif self.branching.has_path(edge.target, edge.source):
            raise ImpossibleMoveException()
#             return self.propose_flip(edge)
        elif self.branching.parent(edge.target) is not None:
            return self.propose_swapping_parent(edge)
        else:
            return self.propose_adding_edge(edge)
        

    def finalize(self):
        '''Update the statistic values after the last iteration.'''
        self.last_tag = np.zeros((len(self.lexicon), len(self.tagset)))
        for i in range(len(self.lexicon)):
            self.last_tag[i, self.current_tag[i]] = 1
        T = len(self.tagset)
        self.tag_freq = \
            self.tag_freq * \
            np.tile(self.last_modified / self.iter_num, (T, 1)).T + \
            self.last_tag * \
                np.tile((self.iter_num-self.last_modified) / self.iter_num,
                        (T, 1)).T
        for stat in self.stats.values():
            stat.update()

    def propose_retagging_node(self, node :LexiconEntry, tag :Tuple[str]) \
                              -> MCMCTagSamplerMove:
        move = MCMCTagSamplerMove(self)
        move.retag_node(node, tag)
        return move

    def propose_deleting_edge(self, edge :GraphEdge) -> MCMCTagSamplerMove:
        move = MCMCTagSamplerMove(self)
        move.remove_edge(edge)
        return move

    def propose_adding_edge(self, edge :GraphEdge) -> MCMCTagSamplerMove:
        move = MCMCTagSamplerMove(self)
        move.add_edge(edge)
        return move

    def propose_swapping_parent(self, edge :GraphEdge) -> MCMCTagSamplerMove:
        move = MCMCTagSamplerMove(self)
        for ingoing_edge in self.branching.ingoing_edges(edge.target):
            move.remove_edge(ingoing_edge)
        move.add_edge(edge)
        return move

    def propose_flip(self, edge :GraphEdge) -> MCMCTagSamplerMove:
        raise NotImplementedError()

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
                            str(edge.rule), str(self.edge_cost_cache[idx])) +\
                            tuple([stat.val[idx] for stat in stats]))

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

class MCMCImprovedTagSampler:

    def __init__(self, full_graph :FullGraph,
                       model :ModelSuite,
                       tagset :List[Tuple[str]],
                       warmup_iter :int = 1000,
                       sampling_iter :int = 100000,
                       iter_stat_interval :int = 1):
        # TODO
        self.lexicon = full_graph.lexicon
        self.model = model
        self.tagset = tagset
        self.tag_idx = { tag : i for i, tag in enumerate(tagset) }
        self.warmup_iter = warmup_iter
        self.sampling_iter = sampling_iter
        self.iter_stat_interval = iter_stat_interval
        self.stats = {}
        self.root_prob = \
            np.zeros((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        logging.getLogger('main').info('Computing root probabilities...')
        for w_id, entry in tqdm.tqdm(enumerate(self.lexicon),
                                     total=len(self.lexicon)):
            for t_id, tag in enumerate(tagset):
                entry_with_tag = \
                    LexiconEntry(''.join(entry.word) + ''.join(tag))
                self.root_prob[w_id,t_id] = \
                    np.exp(-self.model.root_cost(entry_with_tag))
#             if not np.any(self.root_prob[w_id,:] > 0):
#                 raise Exception('Zero root prob: {}'.format(self.lexicon[w_id]))
        logging.getLogger('main').info('Computing leaf probabilities...')
        self.fast_compute_leaf_prob(self.lexicon, model.rule_set)
        logging.getLogger('main').info('Computing transition matrices...')
        untagged_edge_set, self.edge_tr_mat = \
            self.compute_untagged_edges_and_transition_mat(full_graph)
        self.full_graph = FullGraph(self.lexicon, untagged_edge_set)
        self.edge_set = self.full_graph.edge_set
        self.init_forward_prob()
        self.init_backward_prob()
#         self.write_debug_info()

    def fast_compute_leaf_prob(self, lexicon, rule_set):
        self.leaf_prob = np.ones((len(self.lexicon), len(self.tagset)))   # ;-)

    def compute_leaf_prob(self, lexicon, rule_set):

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

        lexicon_tr = lexicon.to_fst()
        lexicon_tr.concatenate(algorithms.fst.generator(self.tagset))
        rules_tr = rule_set.to_fst()
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

    def compute_untagged_edges_and_transition_mat(self, full_graph):

        def _untag_edge(lexicon, edge):
            source = lexicon.get_by_symstr(''.join(edge.source.word))[0]
            target = lexicon.get_by_symstr(''.join(edge.target.word))[0]
            rule = Rule(edge.rule.subst)
            return GraphEdge(source, target, rule)

        edge_prob = self.model.edges_prob(full_graph.edge_set)
        edge_prob_ratios = edge_prob / (1-edge_prob)
        untagged_edge_set = EdgeSet(full_graph.lexicon)
        T = len(self.tagset)
        edge_tr_mat = []
        for e_id, edge in enumerate(full_graph.edge_set):
            untagged_edge = _untag_edge(self.lexicon, edge)
            if untagged_edge not in untagged_edge_set:
                untagged_edge_set.add(untagged_edge)
                edge_tr_mat.append(csr_matrix((T, T)))
            ue_id = untagged_edge_set.get_id(untagged_edge)
            t1_id = self.tag_idx[edge.rule.tag_subst[0]]
            t2_id = self.tag_idx[edge.rule.tag_subst[1]]
            edge_tr_mat[ue_id][t1_id, t2_id] = edge_prob_ratios[e_id]
        return untagged_edge_set, edge_tr_mat

    def init_forward_prob(self):
        self.forward_prob = \
            np.empty((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        for w_id in range(len(self.lexicon)):
            self.forward_prob[w_id,:] = self.root_prob[w_id,:]
#             if not np.any(self.forward_prob[w_id,:] > 0):
#                 raise Exception('Zero root prob: {}'.format(self.lexicon[w_id]))

    def init_backward_prob(self):
        self.backward_prob = \
            np.empty((len(self.lexicon), len(self.tagset)), dtype=np.float64)
        for w_id in range(len(self.lexicon)):
            self.backward_prob[w_id,:] = self.leaf_prob[w_id,:]

    def add_stat(self, name: str, stat :MCMCStatistic) -> None:
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def reset(self):
        self.iter_num = 0
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

    def next(self) -> None:
        # increase the number of iterations
        self.iter_num += 1

        # select an edge randomly
        edge = self.full_graph.random_edge()

        # try the move determined by the selected edge
        try:
            if not np.any(self.root_prob[self.lexicon.get_id(edge.source)] > 0) \
                    or not np.any(self.root_prob[self.lexicon.get_id(edge.target)] > 0):
                raise ImpossibleMoveException
            edges_to_add, edges_to_remove, acc_prob =\
                self.determine_move_proposal(edge)
#             for edge in edges_to_add:
#                 logging.getLogger('main').info('Adding: {}'.format(edge))
#             for edge in edges_to_remove:
#                 logging.getLogger('main').info('Removing: {}'.format(edge))
#             print('acc_prob =', acc_prob)
            if np.isnan(acc_prob):
                raise Exception()
            if acc_prob >= 1 or acc_prob >= random.random():
                self.accept_move(edges_to_add, edges_to_remove)
#                 logging.getLogger('main').info('accepted')
#             else:
#                 logging.getLogger('main').info('rejected')
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
#             return self.propose_flip(edge)
            raise ImpossibleMoveException()
        elif self.branching.parent(edge.target) is not None:
            return self.propose_swapping_parent(edge)
        else:
            return self.propose_adding_edge(edge)

    def check_edge(self, edge :GraphEdge):
#         return True
        if self.branching.degree(edge.source) >= 10:
            return False
        if self.branching.depth(edge.source) + self.branching.height(edge.target) >= 10:
            return False
        return True

    def propose_adding_edge(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        if not self.check_edge(edge):
            raise ImpossibleMoveException()
        src_id = self.lexicon.get_id(edge.source)
        tgt_id = self.lexicon.get_id(edge.target)
        e_id = self.full_graph.edge_set.get_id(edge)
        tr_mat = self.edge_tr_mat[e_id]
        new_src_backward_prob = \
            self.backward_prob[src_id] * tr_mat.dot(self.backward_prob[tgt_id])
#         print(self.forward_prob[src_id])
#         print(self.backward_prob[src_id])
#         print(self.forward_prob[tgt_id])
#         print(self.backward_prob[tgt_id])
#         print(new_src_backward_prob)
#         print()
#         print(np.sum(self.forward_prob[src_id]*self.backward_prob[src_id]))
#         print(np.sum(self.forward_prob[tgt_id]*self.backward_prob[tgt_id]))
#         print(np.sum(self.forward_prob[src_id]*new_src_backward_prob))
        acc_prob = \
            np.sum(self.forward_prob[src_id]*new_src_backward_prob) /\
            (np.sum(self.forward_prob[src_id]*self.backward_prob[src_id]) * \
             np.sum(self.forward_prob[tgt_id]*self.backward_prob[tgt_id]))
        return [edge], [], acc_prob

    def propose_deleting_edge(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        src_id = self.lexicon.get_id(edge.source)
        tgt_id = self.lexicon.get_id(edge.target)
        e_id = self.full_graph.edge_set.get_id(edge)
        tr_mat = self.edge_tr_mat[e_id]
#         print()
#         print('delete')
#         print(self.leaf_prob[src_id,:])
        new_src_backward_prob = np.copy(self.leaf_prob[src_id,:])
        for o_edge in self.branching.outgoing_edges(edge.source):
            if o_edge != edge:
                o_e_id = self.full_graph.edge_set.get_id(o_edge)
                o_tgt_id = self.lexicon.get_id(o_edge.target)
                new_src_backward_prob *= \
                    self.edge_tr_mat[o_e_id].dot(self.backward_prob[o_tgt_id,:])
        acc_prob = \
            np.sum(self.root_prob[tgt_id]*self.backward_prob[tgt_id]) * \
            np.sum(self.forward_prob[src_id]*new_src_backward_prob) /\
            np.sum(self.forward_prob[src_id]*self.backward_prob[src_id])
#         print(self.leaf_prob[src_id,:])
#         print()
        return [], [edge], acc_prob

    def propose_swapping_parent(self, edge :GraphEdge) \
                             -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        if not self.check_edge(edge):
            raise ImpossibleMoveException()
        edge_to_remove = self.branching.edges_between(
                              self.branching.parent(edge.target),
                              edge.target)[0]
        src_id = self.lexicon.get_id(edge.source)
        src_2_id = self.lexicon.get_id(edge_to_remove.source)
        tgt_id = self.lexicon.get_id(edge.target)
        e_id = self.full_graph.edge_set.get_id(edge)
        e2_id = self.full_graph.edge_set.get_id(edge_to_remove)
        tr_mat = self.edge_tr_mat[e_id]
#         print()
#         print('swap')
#         print(self.leaf_prob[src_2_id,:])
        new_src_backward_prob = \
            self.backward_prob[src_id] * tr_mat.dot(self.backward_prob[tgt_id])
        new_src_2_backward_prob = np.copy(self.leaf_prob[src_2_id,:])
        for o_edge in self.branching.outgoing_edges(edge_to_remove.source):
            if o_edge != edge_to_remove:
                o_e_id = self.full_graph.edge_set.get_id(o_edge)
                o_tgt_id = self.lexicon.get_id(o_edge.target)
                new_src_2_backward_prob *= \
                    self.edge_tr_mat[o_e_id].dot(self.backward_prob[o_tgt_id,:])
        # TODO do not divide!!! recompute instead
#         new_src_backward_prob = self.leaf_prob[src_id,:]
#         for outgoing_edge in self.branching.outgoing_edges(edge.source):
#             if outgoing_edge.target != edge.target:
#         new_src_backward_prob = \
#             self.backward_prob[src_id] * tr_mat.dot(self.backward_prob[tgt_id]) /\
#             tr_mat_2.dot(self.backward_prob[tgt_id])
#         print(self.forward_prob[src_id])
#         print(self.backward_prob[src_id])
#         print(self.forward_prob[tgt_id])
#         print(self.backward_prob[tgt_id])
#         print(new_src_backward_prob)
#         print()
#         print(np.sum(self.forward_prob[src_id]*self.backward_prob[src_id]))
#         print(np.sum(self.forward_prob[tgt_id]*self.backward_prob[tgt_id]))
#         print(np.sum(self.forward_prob[src_id]*new_src_backward_prob))
#         print(self.forward_prob[src_id]*new_src_backward_prob)
#         print(self.forward_prob[src_id]*self.backward_prob[src_id])
#         print(self.forward_prob[src_2_id]*new_src_2_backward_prob)
#         print(self.forward_prob[src_2_id]*self.backward_prob[src_2_id])
#         print(self.leaf_prob[src_2_id,:])
#         print()
        acc_prob = \
            np.sum(self.forward_prob[src_id]*new_src_backward_prob) /\
            np.sum(self.forward_prob[src_id]*self.backward_prob[src_id]) * \
            np.sum(self.forward_prob[src_2_id]*new_src_2_backward_prob) /\
            np.sum(self.forward_prob[src_2_id]*self.backward_prob[src_2_id])
        return [edge], [edge_to_remove], acc_prob

#     def compute_acc_prob(self, edges_to_add :List[GraphEdge], 
#                          edges_to_remove :List[GraphEdge], 
#                          prop_prob_ratio :float) -> float:
# 
#         pass
# 
#         for edge in edges_to_add:
#             print('Adding:', edge)
#         for edge in edges_to_remove:
#             print('Removing:', edge)
#         print('acc_prob =', acc_prob)
#         return acc_prob

    def recompute_forward_prob_for_node(self, node):
        w_id = self.lexicon.get_id(node)
        if self.branching.parent(node) is None:
            result = np.copy(self.root_prob[w_id,:])
        else:
            parent = self.branching.parent(node)
            par_id = self.lexicon.get_id(parent)
            result = np.copy(self.forward_prob[par_id,:])
#             print(result)
            for edge in self.branching.outgoing_edges(parent):
                if edge.target != node:
                    e_id = self.full_graph.edge_set.get_id(edge)
                    w2_id = self.lexicon.get_id(edge.target)
#                     print()
#                     print(edge)
#                     print(self.edge_tr_mat[e_id].toarray())
#                     print(self.backward_prob[w2_id,:])
                    result *= self.edge_tr_mat[e_id].dot(self.backward_prob[w2_id,:])
#                     print(result)
#                     print()
            e_id = self.full_graph.edge_set.get_id(\
                       self.branching.ingoing_edges(node)[0])
#             print(result)
#             print(self.edge_tr_mat[e_id].toarray())
            result = np.dot(result, self.edge_tr_mat[e_id].toarray())
        self.forward_prob[w_id,:] = result
        if not np.any(result > 0):
            raise Exception('Zero forward prob. for: {}'.format(self.lexicon[w_id]))

    def recompute_forward_prob_for_subtree(self, root):
        self.recompute_forward_prob_for_node(root)
        for node in self.branching.successors(root):
            self.recompute_forward_prob_for_subtree(node)

    def recompute_backward_prob_for_node(self, node):
        w_id = self.lexicon.get_id(node)
        result = np.copy(self.leaf_prob[w_id,:])
#         print(node)
#         print(self.leaf_prob[w_id,:])
#         print(result)
        for edge in self.branching.outgoing_edges(node):
            e_id = self.full_graph.edge_set.get_id(edge)
            w2_id = self.lexicon.get_id(edge.target)
            result *= self.edge_tr_mat[e_id].dot(self.backward_prob[w2_id,:])
#             print(edge)
#             print(self.backward_prob[w2_id,:])
#             print(result)
#         print()
        self.backward_prob[w_id,:] = result
        if not np.any(result > 0):
            raise Exception('Zero backward prob. for: {}'.format(self.lexicon[w_id]))

    def recompute_backward_prob_for_subtree(self, root):
        for node in self.branching.successors(root):
            self.recompute_backward_prob_for_subtree(node)
        self.recompute_backward_prob_for_node(root)

    def update_tag_freq_for_node(self, node):
        w_id = self.lexicon.get_id(node)
        cur_tag_freq = self.forward_prob[w_id,:] * self.backward_prob[w_id,:]
        cur_tag_freq_sum = np.sum(cur_tag_freq)
        if cur_tag_freq_sum > 0:
            cur_tag_freq /= np.sum(cur_tag_freq)
        else:
            cur_tag_freq = np.ones(len(self.tagset)) / len(self.tagset)
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
        for root in roots_changed:
            self.update_tag_freq_for_subtree(root)
            self.recompute_backward_prob_for_subtree(root)
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
                edge_tr_mat = self.edge_tr_mat[e_id].toarray()
                for t1_id, tag_1 in enumerate(self.tagset):
                    for t2_id, tag_2 in enumerate(self.tagset):
                        prob = edge_tr_mat[t1_id,t2_id]
                        if prob > 0:
                            tag_probs.append((''.join(tag_1), ''.join(tag_2), str(prob)))
                write_line(fp, (str(edge), ' '.join([t1+':'+t2+':'+prob \
                                                     for t1, t2, prob in tag_probs])))

    def write_debug_info(self):
        self.write_root_prob('sampler-root-prob.txt')
#         self.write_leaf_prob('sampler-leaf-prob.txt')
        self.write_edge_tr_mat('sampler-edge-tr-mat.txt')

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

