import pickle
from abc import ABC

import spacy
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


def _unpack(x):
    return list(map(lambda i: pickle.loads(i), x))


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


class DataLoaderRatebeerBow(ADataLoader):
    def __init__(self, **kwargs):
        batch_size = kwargs.pop('batch_size', 32)
        fix_len = kwargs.pop('fix_len', None)
        bptt_length = kwargs.pop('bptt_len', 20)
        num_features = kwargs.pop('num_features')
        server = kwargs.pop('server', 'localhost')

        FIELD_TIME = data.BPTTField(bptt_len=bptt_length, sequential=True, use_vocab=False, batch_first=True,
                                    include_lengths=True,
                                    fix_length=fix_len, pad_token=0)
        FIELD_TEXT = data.BPTTField(bptt_len=bptt_length, sequential=True, use_vocab=False, batch_first=True,
                                    include_lengths=False,
                                    preprocessing=_unpack,
                                    postprocessing=lambda data, y: [list(map(lambda x: x.toarray()[0], d)) for d in
                                                                    data],
                                    fix_length=fix_len, pad_token=csr_matrix((1, num_features)))

        train, valid, test = datasets.RatebeerBow.splits(server, time_field=FIELD_TIME,
                                                         text_field=FIELD_TEXT, **kwargs)

        if fix_len == -1:
            max_len = max([train.max_len, valid.max_len, test.max_len])
            FIELD_TIME.fix_length = max_len
            FIELD_TEXT.fix_length = max_len

        self._train_iter, self._valid_iter, self._test_iter = data.BPTTIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                bptt_len=bptt_length)
        self.bptt_length = bptt_length

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
