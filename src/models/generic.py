class Model:
    pass


class ModelFactory:
    @staticmethod
    def create(model_type :str) -> Model:
        raise NotImplementedError()

    @staticmethod
    def load(model_type :str, filename :str) -> Model:
        raise NotImplementedError()


class UnknownModelTypeException(Exception):
    def __init__(self, model_class :str, model_type :str) -> None:
        self.value = 'Unknown {} model type: {}'\
                     .format(model_class, model_type)

    def __str__(self) -> str:
        return self.value

