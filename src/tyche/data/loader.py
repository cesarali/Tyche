import pickle
from abc import ABC
from typing import List

import spacy
import torch
from scipy.sparse import csr_matrix
from torch.utils.data.dataloader import DataLoader

from tyche import data
from tyche.data import datasets

spacy_en = spacy.load('en')


def tokenizer(x):
    """
    Create a tokenizer function
    """
    return [tok.text for tok in spacy_en.tokenizer(x) if tok.text != ' ']


def unpack_bow(x):
    return list(map(lambda i, y: [pickle.loads(i), pickle.loads(y)], x[1:], x[2:]))


def unpack_bow2seq(x):
    return list(map(lambda i: pickle.loads(i), x[1:-1]))


def unpack_text(x):
    return x[2:]


def delta(x: list) -> List[list]:
    dt = [x1 - x2 for (x1, x2) in zip(x[1:], x[:-1])]
    dy = dt[1:]
    return [[x1, x2, y] for (x1, x2, y) in zip(x[1:-1], dt, dy)]


def expand_bow_vector(input, y):
    return [list(map(lambda x: x.toarray()[0], d)) for d in input]


class ADataLoader(ABC):

    @property
    def train(self):
        pass

    @property
    def validate(self):
        pass

    @property
    def test(self):
        pass


class DataLoaderRatebeer(ADataLoader):
    def __init__(self, **kwargs):
        batch_size = kwargs.pop('batch_size', 32)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_len', 20)
        bow_size = kwargs.get('bow_size')
        server = kwargs.pop('server', 'localhost')
        data_collection_name = kwargs.pop('data_collection')

        FIELD_TIME = data.BPTTField(bptt_len=bptt_length, use_vocab=False,
                                    include_lengths=True, pad_token=[0, 0, 0],
                                    preprocessing=delta)
        FIELD_BOW = data.BPTTField(bptt_len=bptt_length, use_vocab=False,
                                   include_lengths=False,
                                   pad_token=csr_matrix((1, bow_size)),
                                   preprocessing=unpack_bow2seq, postprocessing=expand_bow_vector,
                                   dtype=torch.float32)

        FIELD_TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='UNK',
                                          tokenize=tokenizer, batch_first=True, use_vocab=True)
        NESTED_TEXT_FIELD = data.NestedBPTTField(FIELD_TEXT, bptt_length=bptt_length, use_vocab=False,
                                                 preprocessing=unpack_text, include_lengths=True)
        train_col = f'{data_collection_name}_train'
        val_col = f'{data_collection_name}_validation'
        test_col = f'{data_collection_name}_test'
        train, valid, test = datasets.RatebeerBow2Seq.splits(server, time_field=FIELD_TIME,
                                                             text_field=NESTED_TEXT_FIELD, bow_field=FIELD_BOW,
                                                             train=train_col, validation=val_col, test=test_col,
                                                             **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            FIELD_TEXT.fix_length = max_len

        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test), batch_sizes=(batch_size, batch_size, len(test)), sort_key=lambda x: len(x.time),
                sort_within_batch=True, repeat=False, bptt_len=bptt_length)
        self.bptt_length = bptt_length
        NESTED_TEXT_FIELD.build_vocab(train, vectors=emb_dim, vectors_cache=path_to_vectors, max_size=voc_size,
                                      min_freq=min_freq)
        self.train_vocab = NESTED_TEXT_FIELD.vocab
        self.fix_length = NESTED_TEXT_FIELD.fix_length

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def fix_len(self):
        return self.fix_length

    @property
    def bptt_len(self):
        return self.bptt_length


class DataLoaderRatebeerBow(ADataLoader):
    def __init__(self, **kwargs):
        batch_size = kwargs.pop('batch_size', 32)
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_len', 20)
        bow_size = kwargs.get('bow_size', 2000)
        server = kwargs.pop('server', 'localhost')

        FIELD_TIME = data.BPTTField(bptt_len=bptt_length, use_vocab=False,
                                    include_lengths=True, pad_token=[0, 0, 0],
                                    preprocessing=delta)
        FIELD_TEXT = data.BPTTField(bptt_len=bptt_length, use_vocab=False,
                                    include_lengths=False,
                                    pad_token=[csr_matrix((1, bow_size)), csr_matrix((1, bow_size))],
                                    preprocessing=unpack_bow, postprocessing=expand_bow_vector,
                                    dtype=torch.float32)

        train_col = 'ratebeer_by_user_train_' + str(bow_size)
        val_col = 'ratebeer_by_user_validation_' + str(bow_size)
        test_col = 'ratebeer_by_user_test_' + str(bow_size)
        train, valid, test = datasets.RatebeerBow.splits(server, time_field=FIELD_TIME, text_field=FIELD_TEXT,
                                                         train=train_col, validation=val_col, test=test_col, **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            FIELD_TEXT.fix_length = max_len

        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test), batch_sizes=(batch_size, batch_size, len(test)), sort_key=lambda x: len(x.time),
                sort_within_batch=True, repeat=False, bptt_len=bptt_length)
        self.bptt_length = bptt_length
        self.bow_s = bow_size

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def fix_len(self):
        return self.fix_length

    @property
    def bptt_len(self):
        return self.bptt_length

    @property
    def bow_size(self):
        return self.bow_s


class DataLoaderPTB(ADataLoader):
    def __init__(self, **kwargs):
        batch_size = kwargs.get('batch_size')
        path_to_data = kwargs.pop('path_to_data')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='UNK', lower=True,
                                    tokenize=tokenizer,
                                    include_lengths=True, fix_length=fix_len, batch_first=True)
        train, valid, test = datasets.PennTreebank.splits(TEXT, root=path_to_data)

        if fix_len == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = data.BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False
        )

        TEXT.build_vocab(train, vectors=emb_dim, vectors_cache=path_to_vectors, max_size=voc_size, min_freq=min_freq)
        self.train_vocab = TEXT.vocab
        self.fix_length = TEXT.fix_length

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self.fix_length


class DataLoaderWiki2(ADataLoader):
    def __init__(self, **kwargs):
        batch_size = kwargs.get('batch_size')
        path_to_data = kwargs.pop('path_to_data')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='UNK',
                                    tokenize=tokenizer,
                                    include_lengths=True, fix_length=fix_len, batch_first=True)
        train, valid, test = datasets.WikiText2.splits(TEXT, root=path_to_data)

        if fix_len == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = data.BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False
        )

        TEXT.build_vocab(train, vectors=emb_dim, vectors_cache=path_to_vectors, max_size=voc_size,
                         min_freq=min_freq)
        self.train_vocab = TEXT.vocab
        self.fix_length = TEXT.fix_length

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self.fix_length


class DataLoaderWiki103(ADataLoader):
    def __init__(self, **kwargs):
        batch_size = kwargs.get('batch_size')
        path_to_data = kwargs.pop('path_to_data')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='UNK',
                                    tokenize=tokenizer,
                                    include_lengths=True, fix_length=fix_len, batch_first=True)
        train, valid, test = datasets.WikiText103.splits(TEXT, root=path_to_data)

        if fix_len == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = data.BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False
        )

        TEXT.build_vocab(train, vectors=emb_dim, vectors_cache=path_to_vectors, max_size=voc_size,
                         min_freq=min_freq)
        self.train_vocab = TEXT.vocab
        self.fix_length = TEXT.fix_length

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def vocab(self):
        return self.train_vocab

    @property
    def fix_len(self):
        return self.fix_length


class BasicEventDataLoader(ADataLoader):
    def __init__(self, **kwargs):
        data_path = kwargs.pop('data_path')
        self.__bptt_size = kwargs.pop('bptt_size')
        train_data = datasets.BasicEventDataset(data_path, bptt_size=self.__bptt_size)
        test_data = datasets.BasicEventDataset(data_path, train=False)
        self.__train_data_loader = DataLoader(train_data, **kwargs)
        self.__test_data_loader = DataLoader(test_data, **kwargs)

    @property
    def train(self):
        return self.__train_data_loader

    @property
    def validate(self):
        return self.__test_data_loader

    @property
    def bptt(self):
        return self.__bptt_size
