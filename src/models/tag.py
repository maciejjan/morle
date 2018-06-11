from models.generic import Model, ModelFactory, UnknownModelTypeException


class TagModel(Model):
    pass


class NGramTagModel(TagModel):
    pass


class RNNTagModel(TagModel):
    pass


class TagModelFactory(ModelFactory):
    @staticmethod
    def create(model_type :str) -> TagModel:
        if model_type == 'none':
            return None
        elif model_type == 'ngram':
            return NGramTagModel()
        elif model_type == 'rnn':
            return RNNTagModel()
        else:
            raise UnknownModelTypeException('root tag', model_type)

    @staticmethod
    def load(model_type :str, filename :str) -> TagModel:
        if model_type == 'none':
            return None
        elif model_type == 'ngram':
            return NGramTagModel.load(filename)
        elif model_type == 'rnn':
            return RNNTagModel.load(filename)
        else:
            raise UnknownModelTypeException('root tag', model_type)

