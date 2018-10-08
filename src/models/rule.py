from datastruct.rules import Rule
from models.generic import Model, ModelFactory, UnknownModelTypeException
from utils.files import open_to_write, read_tsv_file, write_line

from collections import defaultdict
from typing import Iterable
import math


class RuleModel(Model):
    def __init__(self) -> None:
        raise NotImplementedError()

    def rule_cost(self, rule :Rule) -> float:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'RuleModel':
        raise NotImplementedError()


class UnigramRuleModel(RuleModel):
    UNKNOWN = '<UNKN>'

    def __init__(self) -> None:
        self.probs = { UnigramRuleModel.UNKNOWN: 1 }

    def fit(self, rules :Iterable[Rule]) -> None:
        counts = defaultdict(lambda: 0)
        counts[UnigramRuleModel.UNKNOWN] += 1
        total = 1
        for rule in rules:
            for sym in rule.seq():
                counts[sym] += 1
                total += 1
        for sym, count in counts.items():
            self.probs[sym] = count/total

    def rule_prob(self, rule :Rule) -> float:
        result = 1.0
        for sym in rule.seq():
            result *= self.probs[sym] \
                      if sym in self.probs \
                      else self.probs[UnigramRuleModel.UNKNOWN]
        return result

    def rule_cost(self, rule :Rule) -> float:
        return -math.log(self.rule_prob(rule))

    def save(self, filename :str) -> None:
        with open_to_write(filename) as fp:
            for sym, prob in self.probs.items():
                line = (sym[0], sym[1], prob) \
                       if isinstance(sym, tuple) \
                       else (sym, prob)
                write_line(fp, line)

    @staticmethod
    def load(filename :str) -> 'UnigramRuleModel':
        result = UnigramRuleModel()
        result.probs = {}
        for row in read_tsv_file(filename):
            if len(row) == 2:
                result.probs[row[0]] = float(row[1])
            elif len(row) == 3:
                result.probs[(row[0], row[1])] = float(row[2])
            else:
                logging.getLogger('main').warning(\
                    'Cannot parse row: {} in {}'\
                    .format(str(row), filename))


class RuleModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str) -> RuleModel:
        if model_type == 'none':
            return None
        elif model_type == 'unigram':
            return UnigramRuleModel()
        else:
            raise UnknownModelTypeException('rule', model_type)

    @staticmethod
    def load(model_type :str, filename :str) -> RuleModel:
        if model_type == 'none':
            return None
        if model_type == 'unigram':
            return UnigramRuleModel.load(filename)
        else:
            raise UnknownModelTypeException('rule', model_type)
