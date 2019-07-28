from abc import ABC

import spacy
from torch.utils.data.dataloader import DataLoader

from fhggeneral import data
from fhggeneral.data import datasets

spacy_en = spacy.load('en')


def tokenizer(x):
    """
    Create a tokenizer function
    """
    return [tok.text for tok in spacy_en.tokenizer(x) if tok.text != ' ']


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
        bptt_size = kwargs.pop('bptt_size')
        train_data = datasets.BasicEventDataset(data_path, bptt_size=bptt_size)
        test_data = datasets.BasicEventDataset(data_path, train=False)
        self.__train_data_loader = DataLoader(train_data, **kwargs)
        self.__test_data_loader = DataLoader(test_data, **kwargs)

    @property
    def train_data_loader(self):
        return self.__train_data_loader

    @property
    def test_data_loader(self):
        return self.__test_data_loader
