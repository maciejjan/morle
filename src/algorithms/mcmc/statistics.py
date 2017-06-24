from datastruct.graph import GraphEdge
from datastruct.lexicon import LexiconEntry
from datastruct.rules import Rule

from typing import Dict, Tuple

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
        return self.values[iter_num // self.sampler.iter_stat_interval-1]


class CostAtIterationStatistic(IterationStatistic):
    def next_iter(self) -> None:
        if self.sampler.iter_num % self.sampler.iter_stat_interval == 0:
            self.values.append(self.sampler.logl())


class EdgeStatistic(MCMCStatistic):
    def reset(self) -> None:
        # TODO use NumPy arrays as values, not lists!
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
    def update(self) -> None:
        for edge in self.sampler.edge_index:
            if self.sampler.branching.has_edge(edge.source, edge.target, edge.rule):
                # not really removing -- just accounting for the fact that
                # the edge was present in the last graphs
                self.edge_removed(edge)
            else:
                # not really adding -- just accounting for the fact that
                # the edge was absent in the last graphs
                self.edge_added(edge)

    def edge_added(self, edge :GraphEdge) -> None:
        idx = self.sampler.edge_index[edge]
        self.values[idx] =\
            self.values[idx] * self.last_modified[idx] / self.sampler.iter_num
        self.last_modified[idx] = self.sampler.iter_num

    def edge_removed(self, edge :GraphEdge) -> None:
        idx = self.sampler.edge_index[edge]
        self.values[idx] =\
            (self.values[idx] * self.last_modified[idx] +\
             (self.sampler.iter_num - self.last_modified[idx])) /\
            self.sampler.iter_num
        self.last_modified[idx] = self.sampler.iter_num


class UnorderedWordPairStatistic(MCMCStatistic):
    def key_from_edge(edge :GraphEdge) -> Tuple[LexiconEntry, LexiconEntry]:
        if edge.source < edge.target:
            return (edge.source, edge.target)
        else:
            return (edge.target, edge.source)

    def reset(self) -> None:
        self.values = [0] * len(self.sampler.unordered_word_pair_index)
        self.last_modified = [0] * len(self.sampler.unordered_word_pair_index)
    
    def update(self) -> None:
        raise NotImplementedError()

    def edge_added(self, edge :GraphEdge) -> None:
        raise NotImplementedError()

    def edge_removed(self, edge :GraphEdge) -> None:
        raise NotImplementedError()

    def next_iter(self):
        pass
    
    def value(self, key :Tuple[LexiconEntry, LexiconEntry]) -> float:
        idx = self.sampler.unordered_word_pair_index[key]
        return self.values[idx]


class UndirectedEdgeFrequencyStatistic(UnorderedWordPairStatistic):
    def update(self) -> None:
        for edge in self.sampler.edge_index:
            if self.sampler.branching.has_edge(\
                        edge.source, edge.target, edge.rule):
                # not really removing -- just accounting for the fact that
                # the edge was present in the last graphs
                self.edge_removed(edge)
            elif self.sampler.branching.has_edge(edge.source, edge.target) or \
                 self.sampler.branching.has_edge(edge.target, edge.source):
                pass
            else:
                # not really adding -- just accounting for the fact that
                # the edge was absent in the last graphs
                self.edge_added(edge)

    def edge_added(self, edge :GraphEdge) -> None:
        key = UnorderedWordPairStatistic.key_from_edge(edge)
        idx = self.sampler.unordered_word_pair_index[key]
        self.values[idx] =\
            self.values[idx] * self.last_modified[idx] / self.sampler.iter_num
        self.last_modified[idx] = self.sampler.iter_num

    def edge_removed(self, edge :GraphEdge) -> None:
        key = UnorderedWordPairStatistic.key_from_edge(edge)
        idx = self.sampler.unordered_word_pair_index[key]
        self.values[idx] =\
            (self.values[idx] * self.last_modified[idx] +\
             (self.sampler.iter_num - self.last_modified[idx])) /\
            self.sampler.iter_num
        self.last_modified[idx] = self.sampler.iter_num


class RuleStatistic(MCMCStatistic):
    def reset(self) -> None:
        self.values = [0] * len(self.sampler.rule_index)
        self.last_modified = [0] * len(self.sampler.rule_index)
    
    def update(self) -> None:
        for rule in self.sampler.rule_index:
            self.update_rule(rule)
    
    def update_rule(self, rule :Rule) -> None:
        raise NotImplementedError()
    
    def value(self, rule :Rule) -> float:
        return self.values[self.sampler.rule_index[rule]]

    def values_dict(self) -> Dict[Rule, float]:
        result = {} # type: Dict[Rule, float]
        for rule, idx in self.sampler.rule_index.items():
            result[rule] = self.values[idx]
        return result


class RuleFrequencyStatistic(RuleStatistic):
    def update_rule(self, rule :Rule) -> None:
        idx = self.sampler.rule_index[rule]
        self.values[idx] = \
            (self.values[idx] * self.last_modified[idx] +\
             self.current_count[idx] * (self.sampler.iter_num - self.last_modified[idx])) /\
            self.sampler.iter_num
        self.last_modified[idx] = self.sampler.iter_num

    def reset(self) -> None:
        super().reset()
        self.current_count = [0] * len(self.sampler.rule_index)
        for rule in self.sampler.rule_index:
            edges = self.sampler.branching.get_edges_for_rule(rule)
            self.current_count[self.sampler.rule_index[rule]] = len(edges)
    
    def edge_added(self, edge :GraphEdge) -> None:
        self.update_rule(edge.rule)
        self.current_count[self.sampler.rule_index[edge.rule]] += 1

    def edge_removed(self, edge :GraphEdge) -> None:
        self.update_rule(edge.rule)
        self.current_count[self.sampler.rule_index[edge.rule]] -= 1


# TODO include rule cost
class RuleExpectedContributionStatistic(RuleStatistic):
    def update_rule(self, rule :Rule) -> None:
        idx = self.sampler.rule_index[rule]
        if self.last_modified[idx] < self.sampler.iter_num:
            edges = self.sampler.branching.get_edges_for_rule(rule)
            new_value = self.sampler.model.cost_of_change([], edges) -\
                self.sampler.model.rule_cost(rule)
            self.values[idx] = \
                (self.values[idx] * self.last_modified[idx] +\
                 new_value * (self.sampler.iter_num - self.last_modified[idx])) /\
                self.sampler.iter_num
            self.last_modified[idx] = self.sampler.iter_num
    
    def edge_added(self, edge :GraphEdge) -> None:
        self.update_rule(edge.rule)

    def edge_removed(self, edge :GraphEdge) -> None:
        self.update_rule(edge.rule)

