from abc import ABC

import spacy
from torchtext.data.iterator import BucketIterator

from tyche import data
from tyche.data import datasets

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
    def __init__(self, device, **kwargs):
        batch_size = kwargs.get('batch_size')
        path_to_data = kwargs.pop('path_to_data')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)
        min_len = kwargs.pop('min_len', None)
        max_len = kwargs.pop('max_len', None)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', lower=True,
                                    tokenize=tokenizer,
                                    include_lengths=True, fix_length=fix_len, batch_first=True)
        train, valid, test = datasets.PennTreebank.splits(TEXT, root=path_to_data)

        if min_len is not None:
            train.examples = [x for x in train.examples if len(x.text) >= min_len]
            valid.examples = [x for x in valid.examples if len(x.text) >= min_len]
            test.examples = [x for x in test.examples if len(x.text) >= min_len]
        if max_len is not None:
            train.examples = [x for x in train.examples if len(x.text) <= max_len]
            valid.examples = [x for x in valid.examples if len(x.text) <= max_len]
            test.examples = [x for x in test.examples if len(x.text) <= max_len]

        if fix_len == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                device=device
        )

        TEXT.build_vocab(train, vectors=emb_dim, vectors_cache=path_to_vectors,
                         max_size=voc_size, min_freq=min_freq)
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


def _preprocess_wiki(dataset, min_len, max_len):
    if min_len is not None:
        dataset.examples = [sent for sent in dataset.examples if len(sent.text) >= min_len]
    if max_len is not None:
        dataset.examples = [sent for sent in dataset.examples if len(sent.text) <= max_len]
    for sent in dataset.examples:
        words = sent.text
        words = [word for word in words if not word == '@-@']
        sent.text = words
    dataset.examples = [sent for sent in dataset.examples if sent.text.count('=') < 2]
    return dataset


class DataLoaderWiki2(ADataLoader):
    def __init__(self, device, **kwargs):
        batch_size = kwargs.get('batch_size')
        path_to_data = kwargs.pop('path_to_data')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)
        min_len = kwargs.pop('min_len', None)
        max_len = kwargs.pop('max_len', None)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>', unk_token='<unk>',
                                    tokenize=None,
                                    include_lengths=True, fix_length=fix_len, batch_first=True)
        train, valid, test = datasets.WikiText2.splits(TEXT, root=path_to_data)

        for dataset in [train, valid, test]:
            dataset = _preprocess_wiki(dataset, min_len, max_len)

        if fix_len == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = data.BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                device=device

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
    def __init__(self, device, **kwargs):
        batch_size = kwargs.get('batch_size')
        path_to_data = kwargs.pop('path_to_data')
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size', None)
        min_freq = kwargs.pop('min_freq', 1)
        fix_len = kwargs.pop('fix_len', None)
        min_len = kwargs.pop('min_len', None)
        max_len = kwargs.pop('max_len', None)

        # Defining fields
        TEXT = data.ReversibleField(init_token='<sos>', eos_token='<eos>',
                                    tokenize=None,
                                    include_lengths=True, fix_length=fix_len, batch_first=True)
        train, valid, test = datasets.WikiText103.splits(TEXT, root=path_to_data)

        for dataset in [train, valid, test]:
            dataset = _preprocess_wiki(dataset, min_len, max_len)

        if fix_len == -1:
            TEXT.fix_length = max([train.max_len, valid.max_len, test.max_len])

        self._train_iter, self._valid_iter, self._test_iter = data.BucketIterator.splits(
                (train, valid, test),
                batch_sizes=(batch_size, batch_size, len(test)),
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                device=device
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
