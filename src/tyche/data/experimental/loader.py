from abc import ABC, abstractmethod

import spacy
import torch
from torch.utils.data.dataloader import DataLoader
from torchtext.data.utils import get_tokenizer

from tyche.data.experimental.datasets import WikiText2, WikiText103, PennTreebank

sampler = torch.utils.data.RandomSampler
DistributedSampler = torch.utils.data.distributed.DistributedSampler

spacy_en = spacy.load('en_core_web_sm')
tokenizer = get_tokenizer("spacy")
URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}


def tokenizer(x, punct=True):
    """
    Create a tokenizer function
    """
    if punct:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_space]
    else:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_punct | token.is_space]


def tokenizer_ptb(x, punct=True):
    """
    Create a tokenizer function and replaces unk tokens in source textfile
    """
    x = x.replace("<unk>", "unk")
    if punct:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_space]
    else:
        return [token.orth_ for token in spacy_en.tokenizer(x) if not token.is_punct | token.is_space]


class ADataLoader(ABC):
    def __init__(self, device, rank: int = 0, world_size: int = -1, **kwargs):
        self.device = device
        self.batch_size = kwargs.pop('batch_size')
        self.path_to_vectors = kwargs.pop('path_to_vectors', None)
        self.emb_dim = kwargs.pop('emb_dim', None)
        self.voc_size = kwargs.pop('voc_size', None)
        self.min_freq = kwargs.pop('min_freq', 1)
        self._fix_length = kwargs.pop('fix_len', None)
        self.min_len = kwargs.pop('min_len', None)
        self.max_len = kwargs.pop('max_len', None)
        self.lower = kwargs.pop('lower', False)
        self.punctuation = kwargs.pop('punctuation', True)
        self.dataset_kwargs = kwargs
        self.world_size = world_size
        self.rank = rank

    @property
    @abstractmethod
    def train(self): ...

    @property
    @abstractmethod
    def validate(self): ...

    @property
    @abstractmethod
    def test(self): ...

    @property
    def n_train_batches(self):
        return len(self.train.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_validate_batches(self):
        return len(self.validate.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_test_batches(self):
        return len(self.test.dataset) // self.batch_size // abs(self.world_size)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)


class DataLoaderPTB(ADataLoader):
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        fix_len = kwargs.pop('fix_len')
        train_dataset, test_dataset, valid_dataset = PennTreebank(root=path_to_data, tokenizer=tokenizer,
                                                                  path_to_vectors=path_to_vectors,
                                                                  emb_dim=emb_dim,
                                                                  voc_size=voc_size,
                                                                  min_freq=min_freq,
                                                                  fix_len=fix_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        vocab = train_dataset.get_vocab()
        self.train_vocab = vocab
        self._fix_length = fix_len

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
        return self._fix_length


class DataLoaderWiki2(ADataLoader):

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        fix_len = kwargs.pop('fix_len')
        train_dataset, test_dataset, valid_dataset = WikiText2(root=path_to_data, tokenizer=tokenizer, path_to_vectors=path_to_vectors,
                                                               emb_dim=emb_dim,
                                                               voc_size=voc_size,
                                                               min_freq=min_freq,
                                                               fix_len=fix_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        vocab = train_dataset.get_vocab()
        self.train_vocab = vocab
        self._fix_length = fix_len

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
        return self._fix_length


class DataLoaderWiki103(ADataLoader):

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        fix_len = kwargs.pop('fix_len')

        train_dataset, test_dataset, valid_dataset = WikiText103(root=path_to_data, tokenizer=tokenizer, path_to_vectors=path_to_vectors, emb_dim=emb_dim,
                                                                 voc_size=voc_size, min_freq=min_freq, fix_len=fix_len)

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        vocab = train_dataset.get_vocab()
        vocab.load_vectors(self.emb_dim, unk_init=None, cache=self.path_to_vectors)
        self.train_vocab = vocab
        # self._fix_length = TEXT.fix_length

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
        return self._fix_length
