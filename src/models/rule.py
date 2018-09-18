from datastruct.rules import Rule
from models.generic import Model, ModelFactory, UnknownModelTypeException

from typing import Iterable


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
    def __init__(self) -> None:
        pass
#         raise NotImplementedError()

    def fit(self, rules :Iterable[Rule]) -> None:
        pass
#         raise NotImplementedError()

    def rule_cost(self, rule :Rule) -> float:
        return 0
#         raise NotImplementedError()

    def save(self, filename :str) -> None:
        pass
#         raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'UnigramRuleModel':
        return UnigramRuleModel()
#         raise NotImplementedError()


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
