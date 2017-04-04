# from algorithms.mcmc.samplers import MCMCGraphSampler
from datastruct.graph import GraphEdge


class MCMCStatistic:
    def __init__(self, sampler :'MCMCGraphSampler') -> None:
        self.sampler = sampler
        self.reset()
    
    def reset(self) -> None:
        pass

    def update(self) -> None:
        pass

    def edge_added(self, edge :GraphEdge) -> None:
        pass

    def edge_removed(self, edge :GraphEdge) -> None:
        pass

    def next_iter(self) -> None:
        pass


class ScalarStatistic(MCMCStatistic):
    def __init__(self, sampler :'MCMCGraphSampler') -> None:
        super().__init__(sampler)
    
    def reset(self) -> None:
        super().reset()
        self.val = 0                # type: float
        self.last_modified = 0      # type: int

    def update(self) -> None:
        raise NotImplementedError()

    def edge_added(self, edge :GraphEdge) -> None:
        raise NotImplementedError()

    def edge_removed(self, edge :GraphEdge) -> None:
        raise NotImplementedError()

    def value(self) -> float:
        return self.val


class ExpectedCostStatistic(ScalarStatistic):
    def __init__(self, sampler :'MCMCGraphSampler') -> None:
        super().__init__(sampler)

    def update(self) -> None:
        pass
    
    def edge_added(self, edge :GraphEdge) -> None:
        pass

    def edge_removed(self, edge :GraphEdge) -> None:
        pass
    
    def next_iter(self):
        self.val = \
            (self.val * (self.sampler.iter_num-1) + self.sampler.logl()) \
            / self.sampler.iter_num


class TimeStatistic(ScalarStatistic):
    def reset(self, sampler :'MCMCGraphSampler') -> None:
        self.started = time.time()
        self.val = 0
    
    def update(self):
        self.val = time.time() - self.started
    
    def edge_added(self, edge :GraphEdge) -> None:
        pass

    def edge_removed(self, edge :GraphEdge) -> None:
        pass
    
    def next_iter(self):
        pass

# TODO deprecated
# class GraphsWithoutRuleSetStatistic(ScalarStatistic):
#     def __init__(self, sampler, forbidden_rules):
#         self.forbidden_rules = forbidden_rules
#         self.reset(sampler)
# 
#     def reset(self, sampler):
#         self.val = 0
#         self.last_modified = 0
#         self.forbidden_edges =\
#             sum(count for rule, count in sampler.lexicon.rules_c.items()\
#                 if rule in self.forbidden_rules)
# 
#     def update(self, sampler):
#         self.val =\
#             (self.val * self.last_modified +\
#              int(self.forbidden_edges == 0) *\
#                 (sampler.num - self.last_modified)) /\
#             sampler.num
#         self.last_modified = sampler.num
# 
#     def edge_added(self, sampler, idx, edge):
#         if edge.rule in self.forbidden_rules:
#             if self.forbidden_edges == 0:
#                 self.update(sampler)
#             self.forbidden_edges += 1
# 
#     def edge_removed(self, sampler, idx, edge):
#         if edge.rule in self.forbidden_rules:
#             if self.forbidden_edges == 1:
#                 self.update(sampler)
#             self.forbidden_edges -= 1
# 
#     def next_iter(self, sampler):
#         pass


class AcceptanceRateStatistic(ScalarStatistic):
    def update(self):
        pass
    
    def edge_added(self, edge :GraphEdge) -> None:
        self.acceptance()

    def edge_removed(self, edge :GraphEdge) -> None:
        self.acceptance()

    def acceptance(self) -> None:
        if self.sampler.iter_num > self.last_modified:
            self.val = (self.val * self.last_modified + 1) / \
                       self.sampler.iter_num
            self.last_modified = self.sampler.iter_num


class IterationStatistic(MCMCStatistic):
    def reset(self) -> None:
        self.values = []        # type: List[float]

    def value(self, iter_num :int) -> float:
        if iter_num % self.sampler.iter_stat_interval != 0:
            raise KeyError(iter_num)
        return self.values[iter_num // self.sampler.iter_stat_interval]


class CostAtIterationStatistic(IterationStatistic):
    def next_iter(self) -> None:
        if self.sampler.iter_num % self.sampler.iter_stat_interval == 0:
            self.values.append(self.sampler.logl())


class EdgeStatistic(MCMCStatistic):
    def reset(self) -> None:
        self.values = [0] * len(self.sampler.edge_index)
        self.last_modified = [0] * len(self.sampler.edge_index)
    
    def update(self) -> None:
        raise NotImplementedError()

    def edge_added(self, edge :GraphEdge) -> None:
        raise NotImplementedError()

    def edge_removed(self, edge :GraphEdge) -> None:
        raise NotImplementedError()

    def next_iter(self):
        pass
    
    def value(self, edge :GraphEdge) -> float:
        return self.values[self.sampler.edge_index[edge]]


class EdgeFrequencyStatistic(EdgeStatistic):
    def update(self):
        for i, edge in enumerate(sampler.edges):
            if edge in edge.source.edges:
                # the edge was present in the last graphs
                self.edge_removed(sampler, i, edge)
            else:
                # the edge was absent in the last graphs
                self.edge_added(sampler, i, edge)

    def edge_added(self, sampler, idx, edge):
        self.values[idx] =\
            self.values[idx] * self.last_modified[idx] / sampler.num
        self.last_modified[idx] = sampler.num

    def edge_removed(self, sampler, idx, edge):
        self.values[idx] =\
            (self.values[idx] * self.last_modified[idx] +\
             (sampler.num - self.last_modified[idx])) /\
            sampler.num
        self.last_modified[idx] = sampler.num


class WordpairStatistic(MCMCStatistic):
    def __init__(self, sampler):
        self.word_ids = {}
        self.words = []
        cur_id = 0
        for node in sampler.lexicon.iter_nodes():
            self.word_ids[node.key] = cur_id
            self.words.append(node.key)
            cur_id += 1
        self.reset(sampler)

    def reset(self, sampler):
        self.values = {}
        self.last_modified = {}
#         self.values = scipy.sparse.lil_matrix(
#                         (len(self.words), len(self.words)), dtype=np.float32)
#         self.last_modified = scipy.sparse.lil_matrix(
#                                (len(self.words), len(self.words)), 
#                                dtype=np.uint32)
        
    def update(self):
        raise NotImplementedError()

    def edge_added(self, sampler, idx, edge):
        raise NotImplementedError()

    def edge_removed(self, sampler, idx, edge):
        raise NotImplementedError()

    def next_iter(self):
        pass

    def value(self, word_1, word_2):
        key = self.key_for_wordpair(word_1, word_2)
        if key in self.values:
            return self.values[key]
        else:
            return 0.0

    def key_for_edge(self, edge):
        key_1 = self.word_ids[edge.source.key]
        key_2 = self.word_ids[edge.target.key]
        return (min(key_1, key_2), max(key_1, key_2))

    def key_for_wordpair(self, word_1, word_2):
        key_1 = self.word_ids[word_1]
        key_2 = self.word_ids[word_2]
        return (min(key_1, key_2), max(key_1, key_2))


class UndirectedEdgeFrequencyStatistic(WordpairStatistic):
    def update(self):
        # note: the relation edge <-> wordpair is one-to-one here because 
        # in well-formed graphs there can only be one edge per wordpair
        for i, edge in enumerate(sampler.edges):
            key = self.key_for_edge(edge)   # only for debug
            if edge in edge.source.edges:
                # the edge was present in the last graphs
                self.edge_removed(sampler, i, edge)
#                 logging.getLogger('main').debug('updating +: %s -> %s : %f' %\
#                     (edge.source.key, edge.target.key, self.values[key]))
        # second loop because all present edges must be processed first
        for i, edge in enumerate(sampler.edges):
            key = self.key_for_edge(edge)   # only for debug
            if edge not in edge.source.edges:
                # the edge was absent in the last graphs
                self.edge_added(sampler, i, edge)
#                 logging.getLogger('main').debug('updating -: %s -> %s : %f' %\
#                     (edge.source.key, edge.target.key, self.values[key]))

    def edge_added(self, sampler, idx, edge):
        key = self.key_for_edge(edge)
        if key not in self.values:
            self.values[key] = 0
            self.last_modified[key] = 0
        self.values[key] =\
            self.values[key] * self.last_modified[key] / sampler.num
        self.last_modified[key] = sampler.num

    def edge_removed(self, sampler, idx, edge):
        key = self.key_for_edge(edge)
        if key not in self.values:
            self.values[key] = 0
            self.last_modified[key] = 0
        self.values[key] =\
            (self.values[key] * self.last_modified[key] +\
             (sampler.num - self.last_modified[key])) /\
            sampler.num
        self.last_modified[key] = sampler.num

# TODO deprecated
# class PathFrequencyStatistic(WordpairStatistic):
#     def reset(self, sampler):
#         WordpairStatistic.reset(self, sampler)
#         self.comp = [None] * len(sampler.lexicon)
#         for root in sampler.lexicon.roots:
#             comp = [self.word_ids[node.key] for node in root.subtree()]
#             for x in comp:
#                 self.comp[x] = comp
#             for x in comp:
#                 for y in comp:
#                     if x != y:
#                         key = (min(x, y), max(x, y))
#                         self.values[key] = 0
#                         self.last_modified[key] = 0
# 
#     def update(self, sampler):
#         for key in self.values:
#             key_1, key_2 = key
#             if self.comp[key_1] == self.comp[key_2]:
#                 self.values[key] =\
#                     (self.values[key] * self.last_modified[key] +\
#                      (sampler.num - self.last_modified[key])) /\
#                     sampler.num
#             else:
#                 self.values[key] =\
#                     self.values[key] * self.last_modified[key] / sampler.num
#             self.last_modified[key] = sampler.num
# 
#     def edge_added(self, sampler, idx, edge):
#         for key_1 in self.comp[self.word_ids[edge.source.key]]:
#             for key_2 in self.comp[self.word_ids[edge.target.key]]:
#                 key = (min(key_1, key_2), max(key_1, key_2))
#                 if key not in self.values:
#                     self.values[key] = 0
#                     self.last_modified[key] = 0
#                 self.values[key] =\
#                     self.values[key] * self.last_modified[key] / sampler.num
#                 self.last_modified[key] = sampler.num
#         # join the subtrees
#         comp_joined = self.comp[self.word_ids[edge.source.key]] +\
#                       self.comp[self.word_ids[edge.target.key]]
#         for x in comp_joined:
#             self.comp[x] = comp_joined
# 
#     def edge_removed(self, sampler, idx, edge):
#         comp_target = [self.word_ids[node.key] \
#                        for node in edge.target.subtree()]
#         comp_source = [x for x in self.comp[self.word_ids[edge.source.key]]\
#                          if x not in comp_target]
#         for key_1 in comp_source:
#             for key_2 in comp_target:
#                 key = (min(key_1, key_2), max(key_1, key_2))
#                 if key not in self.values:
#                     self.values[key] = 0
#                     self.last_modified[key] = 0
#                 self.values[key] =\
#                     (self.values[key] * self.last_modified[key] +\
#                      (sampler.num - self.last_modified[key])) /\
#                     sampler.num
#                 self.last_modified[key] = sampler.num
#         # split the component
#         for x in comp_source:
#             self.comp[x] = comp_source
#         for x in comp_target:
#             self.comp[x] = comp_target

# #     def next_iter(self, sampler):
# #         if sampler.num % 1000 == 0:
# #             logging.getLogger('main').debug('size of PathFrequencyStatistic dict: %d' %\
# #                 len(self.values))

class RuleStatistic(MCMCStatistic):
    def __init__(self, sampler):
        self.values = {}
        self.last_modified = {}
        self.reset(sampler)

    def reset(self, sampler):
#        for rule in sampler.model.rules:
        for rule in sampler.model.rule_features:
            self.values[rule] = 0.0
            self.last_modified[rule] = 0
    
    def update(self):
#        for rule in sampler.model.rules:
        for rule in sampler.model.rule_features:
            self.update_rule(rule, sampler)
    
    def update_rule(self, rule, sampler):
        raise Exception('Not implemented!')
    
    def edge_added(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def edge_removed(self, sampler, idx, edge):
        raise Exception('Not implemented!')

    def next_iter(self):
        pass
    
    def value(self, rule):
        return self.values[rule]


class RuleFrequencyStatistic(RuleStatistic):
    def update_rule(self, rule, sampler):
        self.values[rule] = \
            (self.values[rule] * self.last_modified[rule] +\
             sampler.lexicon.rules_c[rule] * (sampler.num - self.last_modified[rule])) /\
            sampler.num
        self.last_modified[rule] = sampler.num
    
    def edge_added(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)

    def edge_removed(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)


# TODO include rule cost
class RuleExpectedContributionStatistic(RuleStatistic):
    def update_rule(self, rule, sampler):
        if self.last_modified[rule] < sampler.num:
            edges = sampler.lexicon.edges_by_rule[rule]
            new_value = sampler.model.cost_of_change([], edges) #TODO + rule_cost
            self.values[rule] = \
                (self.values[rule] * self.last_modified[rule] +\
                 new_value * (sampler.num - self.last_modified[rule])) /\
                sampler.num
            self.last_modified[rule] = sampler.num
    
    def edge_added(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)

    def edge_removed(self, sampler, idx, edge):
        self.update_rule(edge.rule, sampler)


# TODO
class RuleChangeCountStatistic(RuleStatistic):
    def reset(self, sampler):
        for rule in sampler.lexicon.ruleset.keys():
            self.values[rule] = 0
            self.last_modified[rule] = 0

    def update_rule(self, rule, sampler):
        pass
    
    def edge_added(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 1:
            self.values[rule] += 1

    def edge_removed(self, sampler, idx, word_1, word_2, rule):
        if sampler.lexicon.rules_c[rule] == 0:
            self.values[rule] += 1


# TODO deprecated
# class RuleGraphsWithoutStatistic(RuleStatistic):
#     def update_rule(self, rule, sampler):
#         if sampler.lexicon.rules_c[rule] > 0:
#             self.values[rule] = \
#                 self.values[rule] * self.last_modified[rule] / sampler.num
#             self.last_modified[rule] = sampler.num
#         else:
#             self.values[rule] = \
#                 (self.values[rule] * self.last_modified[rule] +\
#                  sampler.num - self.last_modified[rule]) / sampler.num
#     
#     def edge_added(self, sampler, idx, word_1, word_2, rule):
#         if sampler.lexicon.rules_c[rule] == 1:
#             self.values[rule] = \
#                 (self.values[rule] * self.last_modified[rule] +\
#                  sampler.num - self.last_modified[rule]) / sampler.num
#             self.last_modified[rule] = sampler.num
# 
#     def edge_removed(self, sampler, idx, word_1, word_2, rule):
#         if sampler.lexicon.rules_c[rule] == 0:
#             self.values[rule] = \
#                 self.values[rule] * self.last_modified[rule] / sampler.num
#             self.last_modified[rule] = sampler.num


# TODO deprecated
# class RuleIntervalsWithoutStatistic(MCMCStatistic):
#     def __init__(self, sampler):
#         self.intervals = {}
#         self.int_start = {}
#         self.reset(sampler)
# 
#     def reset(self, sampler):
#         for rule in sampler.lexicon.ruleset.keys():
#             self.intervals[rule] = []
#             if sampler.lexicon.rules_c[rule] > 0:
#                 self.int_start[rule] = None
#             else:
#                 self.int_start[rule] = 0
# 
#     def update(self, sampler):
#         for rule in sampler.lexicon.ruleset.keys():
#             if self.int_start[rule] is not None:
#                 self.intervals[rule].append((self.int_start[rule], sampler.num))
#                 self.int_start[rule] = None
#     
#     def edge_added(self, sampler, idx, word_1, word_2, rule):
#         if sampler.lexicon.rules_c[rule] == 1:
#             if self.int_start[rule] is None:
#                 raise Exception('Interval with no left end: %s' % rule)
#             self.intervals[rule].append((self.int_start[rule], sampler.num))
#             self.int_start[rule] = None
# 
#     def edge_removed(self, sampler, idx, word_1, word_2, rule):
#         if sampler.lexicon.rules_c[rule] == 0:
#             self.int_start[rule] = sampler.num
#     
#     def next_iter(self, sampler):
#         pass
