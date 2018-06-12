from datastruct.lexicon import LexiconEntry, Lexicon
from models.generic import Model, ModelFactory, UnknownModelTypeException
import shared
from utils.files import open_to_read, open_to_write, write_line

from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, SimpleRNN
import keras
import numpy as np
import os.path
from typing import Iterable, Tuple
import yaml


class TagModel(Model):
    def __init__(self, entries :Iterable[LexiconEntry]) -> None:
        raise NotImplementedError()

    def root_cost(self, entry :LexiconEntry) -> float:
        raise NotImplementedError()

    def save(self, filename :str) -> None:
        raise NotImplementedError()

    @staticmethod
    def load(filename :str) -> 'RootModel':
        raise NotImplementedError()


class NGramTagModel(TagModel):
    pass


class RNNTagModel(TagModel):
    def __init__(self) -> None:
        self.nn = None

    def fit(self, lexicon :Lexicon, weights :np.ndarray) -> None:
        if self.nn is None:
            self._set_parameters(lexicon)
            self._compile_network()
        X, y = self._prepare_data(lexicon)
        self.nn.fit(X, y, epochs=5, sample_weight=weights, batch_size=64,
                    verbose=1)

    def root_cost(self, entry :LexiconEntry) -> float:
        return self.root_costs([entry])[0]

    def root_costs(self, lexicon :Lexicon) -> np.ndarray:
        X, y = self._prepare_data(lexicon)
        y_pred = self.nn.predict(X)
        probs = np.empty(y.shape[0])
        for i in range(y.shape[0]):
            probs[i] = y_pred[i,y[i]]
#         return np.log(probs+1e-300)     # avoid zeros -- TODO a more elegant solution
        return np.log(probs)

    def save(self, filename :str) -> None:
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        weights = self.nn.get_weights()
        np.savez(file_full_path, emb=weights[0], rnn_1=weights[1],
                 rnn_2=weights[2], rnn_3=weights[3], d1_1=weights[4],
                 d1_2=weights[5])
        metadata = {\
            'alphabet' : self.alphabet,
            'tagset' : self.tagset,
            'maxlen' : self.maxlen,
            'dim'    : self.dim,
        }
        with open_to_write('root-tag.model.txt') as fp:
            yaml.dump(metadata, fp)
        
#         with open_to_write('tagset.txt') as fp:
#             for tag in self.tagset:
#                 write_line(fp, (''.join(tag),))

    @staticmethod
    def load(filename :str) -> 'RNNTagModel':
        result = RNNTagModel()
        with open_to_read('root-tag.model.txt') as fp:
            metadata = yaml.load(fp)
            result.alphabet = metadata['alphabet']
            result.alphabet_idx = { sym : i for i, sym in enumerate(result.alphabet, 1) }
            result.tagset = metadata['tagset']
            result.tagset_idx = { tag : i for i, tag in enumerate(result.tagset) }
            result.maxlen = metadata['maxlen']
            result.dim = metadata['dim']
        file_full_path = os.path.join(shared.options['working_dir'], filename)
        result._compile_network()
        with np.load(file_full_path) as data:
            result.nn.layers[0].set_weights([data['emb']])
            result.nn.layers[1].set_weights([data['rnn_1'], data['rnn_2'],
                                             data['rnn_3']])
            result.nn.layers[2].set_weights([data['d1_1'], data['d1_2']])
        return result

    def _set_parameters(self, lexicon :Lexicon) -> None:
        self.dim = 10               # hidden layer size
        self.alphabet = lexicon.get_alphabet() # TODO filter -- only non-tag symbols
        self.alphabet_idx = { sym : i for i, sym in enumerate(self.alphabet, 1) }
        self.tagset = lexicon.get_tagset()
        self.tagset_idx = { tag : i for i, tag in enumerate(self.tagset) }
        self.maxlen = lexicon.get_max_word_length()

    def _compile_network(self) -> None:
        self.nn = Sequential()
        self.nn.add(Embedding(input_dim=len(self.alphabet)+1,
                              output_dim=self.dim, mask_zero=True,
                              input_length=self.maxlen))
        self.nn.add(SimpleRNN(self.dim, activation='relu',
                              return_sequences=False))
        self.nn.add(Dense(len(self.tagset), use_bias=True, activation='softmax'))
        self.nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    def _prepare_data(self, entries :Iterable[LexiconEntry]) -> Tuple[np.ndarray, np.ndarray]:
        X_lst, y_lst = [], []
        for entry in entries:
            X_lst.append([(self.alphabet_idx[sym] \
                           if sym in self.alphabet_idx \
                           else 0) \
                          for sym in entry.word])
            y_lst.append(self.tagset_idx[entry.tag])
        X = keras.preprocessing.sequence.pad_sequences(X_lst, maxlen=self.maxlen)
        y = np.array(y_lst)
        return X, y


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

