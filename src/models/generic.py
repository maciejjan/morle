class Model:
    pass


class ModelFactory:
    pass


class UnknownModelTypeException(Exception):
    def __init__(self, model_class :str, model_type :str) -> None:
        self.value = 'Unknown {} model type: {}'\
                     .format(model_class, model_type)

    def __str__(self) -> str:
        return self.value

