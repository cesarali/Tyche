from abc import ABC, abstractmethod

import torch
from nltk.tokenize import TweetTokenizer
from torch.utils.data.dataloader import DataLoader
from tyche.data.experimental.datasets import WikiText2, WikiText103, PennTreebank, YelpReviewPolarity, YelpReviewFull,\
    YahooAnswers, PennTreebankPretrained, YahooAnswersPretrained, WikiText103Pretrained, Atomic2

sampler = torch.utils.data.RandomSampler

DistributedSampler = torch.utils.data.distributed.DistributedSampler

tokenizer = TweetTokenizer(preserve_case=False).tokenize


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
        return len(self.train)

    @property
    def n_validate_batches(self):
        return len(self.validate)

    @property
    def n_test_batches(self):
        return len(self.test)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        return len(self.test.dataset)


class DataLoaderPTB(ADataLoader):
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        min_len = kwargs.pop('min_len')
        fix_len = kwargs.pop('fix_len')
        train_dataset, test_dataset, valid_dataset = PennTreebank(root=path_to_data, tokenizer=tokenizer,
                                                                  path_to_vectors=path_to_vectors,
                                                                  emb_dim=emb_dim,
                                                                  voc_size=voc_size,
                                                                  min_freq=min_freq,
                                                                  fix_len=fix_len,
                                                                  min_len=min_len)

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
        min_len = kwargs.pop('min_len')
        train_dataset, test_dataset, valid_dataset = WikiText2(root=path_to_data, tokenizer=tokenizer, path_to_vectors=path_to_vectors,
                                                               emb_dim=emb_dim,
                                                               voc_size=voc_size,
                                                               min_freq=min_freq,
                                                               fix_len=fix_len,
                                                               min_len=min_len)

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
        min_len = kwargs.pop('min_len')

        train_dataset, test_dataset, valid_dataset = WikiText103(root=path_to_data, tokenizer=tokenizer, path_to_vectors=path_to_vectors, emb_dim=emb_dim,
                                                                 voc_size=voc_size, min_freq=min_freq, fix_len=fix_len, min_len=min_len)
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
        vocab.load_vectors(self.emb_dim, unk_init=None, cache=self.path_to_vectors)
        self.train_vocab = vocab

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


class DataLoaderYelpReviewPolarity(ADataLoader):
    """
    This is a data loader for the Yelp review dataset with sentiment. The user can also change the proportion of visible target labels
    i.e. from fully supervised to un-supervised.
    """

    def __init__(self, device, rank: int = 0, world_size=-1, supervised_proportion=0.9, **kwargs):
        """
        Provide the location folder of the dataset, if the folder is empty or does not exists the data will be downloaded automatically.
        :param device:
        :param rank:
        :param world_size:
        :param supervised_proportion: the proportion of the data with masked labels,
        :param kwargs:
        """
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        fix_len = kwargs.pop('fix_len')
        min_len = kwargs.pop('min_len')

        train_dataset, test_dataset, valid_dataset = YelpReviewPolarity(root=path_to_data, tokenizer=tokenizer, path_to_vectors=path_to_vectors,
                                                                        emb_dim=emb_dim,
                                                                        voc_size=voc_size, min_freq=min_freq, fix_len=fix_len, min_len=min_len,
                                                                        supervised_proportion=supervised_proportion)
        train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.data['weights'], num_samples=len(train_dataset.data['weights']))
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            raise NotImplementedError('Distributed training and semi-supervised data loaders not implemented!')

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        vocab = train_dataset.vocab
        vocab.load_vectors(self.emb_dim, unk_init=None, cache=self.path_to_vectors)
        self.train_vocab = vocab

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


class DataLoaderSemiSupervised(ADataLoader):
    """
    This is a data loader for any labelled text classification data set. The user can also change the proportion of visible target labels
    i.e. from fully supervised to un-supervised.
    """

    def __init__(self, device, rank: int = 0, world_size=-1, supervised_proportion=0.9, **kwargs):
        """
        Provide the location folder of the dataset, if the folder is empty or does not exists the data will be downloaded automatically.
        :param device:
        :param rank:
        :param world_size:
        :param supervised_proportion: the proportion of the data with masked labels,
        :param kwargs:
        :param dataset: [YelpReviewPolarity, YelpReviewFull, YahooAnswers]
        """

        dataset = eval(kwargs.pop('dataset'))
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop('path_to_vectors')
        emb_dim = kwargs.pop('emb_dim')
        voc_size = kwargs.pop('voc_size')
        min_freq = kwargs.pop('min_freq')
        fix_len = kwargs.pop('fix_len')
        min_len = kwargs.pop('min_len')

        train_dataset, test_dataset, valid_dataset = dataset(root=path_to_data, tokenizer=tokenizer, path_to_vectors=path_to_vectors,
                                                                        emb_dim=emb_dim,
                                                                        voc_size=voc_size, min_freq=min_freq, fix_len=fix_len, min_len=min_len,
                                                                        supervised_proportion=supervised_proportion)
        train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.data['weights'], num_samples=len(train_dataset.data['weights']))
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            raise NotImplementedError('Distributed training and semi-supervised data loaders not implemented!')

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        vocab = train_dataset.vocab
        vocab.load_vectors(self.emb_dim, unk_init=None, cache=self.path_to_vectors)
        self.train_vocab = vocab

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


class DataLoaderPTBPretrained(ADataLoader):
    """
    Data loader for PTB with pretrained tokenizers and models from huggingface
    """
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        min_len = kwargs.pop('min_len')
        fix_len = kwargs.pop('fix_len')
        pretrained_tokenizer = kwargs.pop('pretrained_tokenizer', None)
        assert pretrained_tokenizer is not None, 'no pretrained tokenizer specified'

        train_dataset, test_dataset, valid_dataset = PennTreebankPretrained(root=path_to_data,
                                                                            pretrained_tokenizer=pretrained_tokenizer,
                                                                            fix_len=fix_len,
                                                                            min_len=min_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler,
                                      shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler,
                                      shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler,
                                     shuffle=test_sampler is None, **kwargs)
        self._pad_token_id = train_dataset.get_pad_token_id()
        self._fix_length = fix_len
        self._num_added_tokens = train_dataset.get_num_added_tokens()

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
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def fix_len(self):
        return self._fix_length

    @property
    def num_added_tokens(self):
        return self._num_added_tokens

    @property
    def vocab(self): # for compatibility with TextTrainer
        return None


class DataLoaderYahooPretrained(ADataLoader):
    """
    Data loader for YahooAnswers with pretrained tokenizers and models from huggingface
    """
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        min_len = kwargs.pop('min_len')
        fix_len = kwargs.pop('fix_len')
        pretrained_tokenizer = kwargs.pop('pretrained_tokenizer', None)
        assert pretrained_tokenizer is not None, 'no pretrained tokenizer specified'

        train_dataset, test_dataset, valid_dataset = YahooAnswersPretrained(root=path_to_data,
                                                                            pretrained_tokenizer=pretrained_tokenizer,
                                                                            fix_len=fix_len,
                                                                            min_len=min_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler,
                                      shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler,
                                      shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler,
                                     shuffle=test_sampler is None, **kwargs)
        self._pad_token_id = train_dataset.get_pad_token_id()
        self._fix_length = fix_len
        self._num_added_tokens = train_dataset.get_num_added_tokens()

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
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def fix_len(self):
        return self._fix_length

    @property
    def num_added_tokens(self):
        return self._num_added_tokens

    @property
    def vocab(self): # for compatibility with TextTrainer
        return None

class DataLoaderWiki103Pretrained(ADataLoader):
    """
    Data loader for YahooAnswers with pretrained tokenizers and models from huggingface
    """
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        min_len = kwargs.pop('min_len')
        fix_len = kwargs.pop('fix_len')
        pretrained_tokenizer = kwargs.pop('pretrained_tokenizer', None)
        assert pretrained_tokenizer is not None, 'no pretrained tokenizer specified'

        train_dataset, test_dataset, valid_dataset = WikiText103Pretrained(root=path_to_data,
                                                                            pretrained_tokenizer=pretrained_tokenizer,
                                                                            fix_len=fix_len,
                                                                            min_len=min_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler,
                                      shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler,
                                      shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler,
                                     shuffle=test_sampler is None, **kwargs)
        self._pad_token_id = train_dataset.get_pad_token_id()
        self._fix_length = fix_len
        self._num_added_tokens = train_dataset.get_num_added_tokens()

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
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def fix_len(self):
        return self._fix_length

    @property
    def num_added_tokens(self):
        return self._num_added_tokens

    @property
    def vocab(self): # for compatibility with TextTrainer
        return None

class DataLoaderAtomic2(ADataLoader):
    """
    Data loader for YahooAnswers with pretrained tokenizers and models from huggingface
    """
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        min_len = kwargs.pop('min_len')
        fix_len = kwargs.pop('fix_len')
        train_dataset, test_dataset, valid_dataset = Atomic2(root=path_to_data,
                                                                            fix_len=fix_len,
                                                                            min_len=min_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler,
                                      shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler,
                                      shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler,
                                     shuffle=test_sampler is None, **kwargs)
        self._pad_token_id = train_dataset.get_pad_token_id()
        self._fix_length = fix_len
        self._num_added_tokens = train_dataset.get_num_added_tokens()

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
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def fix_len(self):
        return self._fix_length

    @property
    def num_added_tokens(self):
        return self._num_added_tokens

    @property
    def vocab(self): # for compatibility with TextTrainer
        return None




