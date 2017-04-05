from algorithms.mcmc.indexing import Index
from algorithms.mcmc.statistics import \
    MCMCStatistic, ScalarStatistic, IterationStatistic, EdgeStatistic, \
    RuleStatistic, UnorderedWordPairStatistic
from datastruct.lexicon import LexiconEntry
from datastruct.graph import GraphEdge, Branching, FullGraph
from models.generic import Model
from utils.files import open_to_write, write_line
import shared

import logging
import math
from operator import itemgetter
import random
import tqdm
from typing import List, Tuple


class ImpossibleMoveException(Exception):
    pass


# TODO monitor the number of moves from each variant and their acceptance rates!
class MCMCGraphSampler:
    def __init__(self, full_graph :FullGraph, 
                       model :Model, 
                       warmup_iter :int = 1000,
                       sampling_iter :int = 100000,
                       iter_stat_interval :int = 1) -> None:
        self.full_graph = full_graph
        self.model = model
        self.warmup_iter = warmup_iter
        self.sampling_iter = sampling_iter
        self.iter_stat_interval = iter_stat_interval
        self.stats = {}               # type: Dict[str, MCMCStatistic]
        self.iter_num = 0

        self.branching = self.full_graph.random_branching()
        self.model.fit_to_branching(self.branching)

        # indices on edges, rules, wordpairs
        self.edge_index = Index()
        for source, target, rule in self.full_graph.edges_iter(keys=True):
            self.edge_index.add(GraphEdge(source, target, rule))
        self.unordered_word_pair_index = Index()
        for source, target in self.full_graph.edges_iter():
            key = (min(source, target), max(source, target))
            self.unordered_word_pair_index.add(key)
        self.rule_index = Index()
        for rule in self.model.rule_features:
            self.rule_index.add(rule)
    
    def add_stat(self, name: str, stat :MCMCStatistic) -> None:
        if name in self.stats:
            raise Exception('Duplicate statistic name: %s' % name)
        self.stats[name] = stat

    def logl(self) -> float:
        return self.model.cost()

    def run_sampling(self) -> None:
        logging.getLogger('main').info('Warming up the sampler...')
#         pp = progress_printer(self.warmup_iter)
        progressbar = tqdm.tqdm(total=self.warmup_iter)
        while self.iter_num < self.warmup_iter:
            if self.next():
                progressbar.update()
        progressbar.close()
        self.reset()
        logging.getLogger('main').info('Sampling...')
        progressbar = tqdm.tqdm(total=self.sampling_iter)
        while self.iter_num < self.sampling_iter:
            if self.next():
                progressbar.update()
        progressbar.close()
        self.update_stats()

    def next(self) -> bool:
        # increase the number of iterations
        self.iter_num += 1

        # select an edge randomly
#         edge_idx = random.randrange(self.len_edges)
        edge = self.full_graph.random_edge()

        # try the move determined by the selected edge
        try:
            edges_to_add, edges_to_remove, prop_prob_ratio =\
                self.determine_move_proposal(edge)
            acc_prob = self.compute_acc_prob(\
                edges_to_add, edges_to_remove, prop_prob_ratio)
            if acc_prob >= 1 or acc_prob >= random.random():
                self.accept_move(edges_to_add, edges_to_remove)
            for stat in self.stats.values():
                stat.next_iter()
            return True
        # if move impossible -- discard this iteration
        except ImpossibleMoveException:
            self.iter_num -= 1
            return False

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

        edge_3_1 = random.choice(self.full_graph.find_edges(node_3, node_1))
        edge_3_2 = self.branching.find_edges(node_3, node_2)[0] \
                   if self.branching.has_edge(node_3, node_2) else None
        edge_4_1 = self.branching.find_edges(node_4, node_1)[0] \
                   if self.branching.has_edge(node_4, node_1) else None

        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        if edge_4_1 is not None:
            edges_to_remove.append(edge_4_1)
        else: raise Exception('!')
        edges_to_add.append(edge_3_1)
        prop_prob_ratio = (1/len(self.full_graph.find_edges(node_3, node_1))) /\
                          (1/len(self.full_graph.find_edges(node_3, node_2)))

        return edges_to_add, edges_to_remove, prop_prob_ratio

    def propose_flip_2(self, edge :GraphEdge) \
            -> Tuple[List[GraphEdge], List[GraphEdge], float]:
        edges_to_add, edges_to_remove = [], []
        node_1, node_2, node_3, node_4, node_5 = self.nodes_for_flip(edge)

        if not self.full_graph.has_edge(node_3, node_5):
            raise ImpossibleMoveException()

        edge_2_5 = self.branching.find_edges(node_2, node_5)[0] \
                   if self.branching.has_edge(node_2, node_5) else None
        edge_3_2 = self.branching.find_edges(node_3, node_2)[0] \
                   if self.branching.has_edge(node_3, node_2) else None
        edge_3_5 = random.choice(self.full_graph.find_edges(node_3, node_5))

        if edge_2_5 is not None:
            edges_to_remove.append(edge_2_5)
        elif node_2 != node_5: raise Exception('!')     # TODO ???
        if edge_3_2 is not None: edges_to_remove.append(edge_3_2)
        edges_to_add.append(edge_3_5)
        prop_prob_ratio = (1/len(self.full_graph.find_edges(node_3, node_5))) /\
                          (1/len(self.full_graph.find_edges(node_3, node_2)))

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
        edges_to_remove = self.branching.find_edges(
                              self.branching.parent(edge.target),
                              edge.target)
        return [edge], edges_to_remove, 1

    def compute_acc_prob(self, edges_to_add, edges_to_remove, prop_prob_ratio):
        return math.exp(\
                -self.model.cost_of_change(edges_to_add, edges_to_remove)) *\
               prop_prob_ratio

    def accept_move(self, edges_to_add, edges_to_remove):
#            print('Accepted')
        # remove edges and update stats
        for e in edges_to_remove:
            self.branching.remove_edge(e)
            self.model.apply_change([], [e])
            for stat in self.stats.values():
                stat.edge_removed(e)
        # add edges and update stats
        for e in edges_to_add:
            self.branching.add_edge(e)
            self.model.apply_change([e], [])
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
            for edge in self.edge_index:
                write_line(fp, 
                           (str(edge.source), str(edge.target), 
                            str(edge.rule)) + tuple([stat.value(edge)\
                                                     for stat in stats]))

    def save_rule_stats(self, filename):
        stats, stat_names = [], []
        for stat_name, stat in sorted(self.stats.items(), key = itemgetter(0)):
            if isinstance(stat, RuleStatistic):
                stat_names.append(stat_name)
                stats.append(stat)
        with open_to_write(filename) as fp:
            write_line(fp, ('rule', 'domsize') + tuple(stat_names))
            for rule in self.rule_index:
                write_line(fp, (str(rule), self.model.rule_features[rule][0].trials) +\
                               tuple([stat.value(rule) for stat in stats]))

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
            for iter_num in range(0, self.sampling_iter, 
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
    def new(*args):
        if shared.config['General'].getboolean('supervised'):
            return MCMCSupervisedGraphSampler(*args)
        else:
            return MCMCGraphSampler(*args)

