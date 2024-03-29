from abc import ABC, abstractmethod

import torch
from nltk.tokenize import TweetTokenizer
from torch.utils.data.dataloader import DataLoader
from tyche.data.experimental.datasets import (
    WikiText2,
    WikiText103,
    PennTreebank,
    YelpReviewPolarity,
    YelpReviewFull,
    YahooAnswers,
    PennTreebankPretrained,
    YahooAnswersPretrained,
    WikiText103Pretrained,
    WikiText2Pretrained,
    Atomic2,
    YelpReviewPretrained,
    Atomic2020,
    ConceptNet,
    # ConceptNet5,
)

sampler = torch.utils.data.RandomSampler

DistributedSampler = torch.utils.data.distributed.DistributedSampler

tokenizer = TweetTokenizer(preserve_case=False).tokenize


class ADataLoader(ABC):
    def __init__(self, device, rank: int = 0, world_size: int = -1, **kwargs):
        self.device = device
        self.batch_size = kwargs.pop("batch_size")
        self.path_to_vectors = kwargs.pop("path_to_vectors", None)
        self.emb_dim = kwargs.pop("emb_dim", None)
        self.voc_size = kwargs.pop("voc_size", None)
        self.min_freq = kwargs.pop("min_freq", 1)
        self._fix_length = kwargs.pop("fix_len", None)
        self.min_len = kwargs.pop("min_len", None)
        self.max_len = kwargs.pop("max_len", None)
        self.lower = kwargs.pop("lower", False)
        self.punctuation = kwargs.pop("punctuation", True)
        self.dataset_kwargs = kwargs
        self.world_size = world_size
        self.rank = rank

    @property
    @abstractmethod
    def train(self):
        ...

    @property
    @abstractmethod
    def validate(self):
        ...

    @property
    @abstractmethod
    def test(self):
        ...

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
        path_to_data = kwargs.pop("path_to_data")
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop("path_to_vectors")
        emb_dim = kwargs.pop("emb_dim")
        voc_size = kwargs.pop("voc_size")
        min_freq = kwargs.pop("min_freq")
        min_len = kwargs.pop("min_len")
        fix_len = kwargs.pop("fix_len")
        train_dataset, test_dataset, valid_dataset = PennTreebank(
            root=path_to_data,
            tokenizer=tokenizer,
            path_to_vectors=path_to_vectors,
            emb_dim=emb_dim,
            voc_size=voc_size,
            min_freq=min_freq,
            fix_len=fix_len,
            min_len=min_len,
        )

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(
            train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
        )
        self._valid_iter = DataLoader(
            valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
        )
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
        path_to_data = kwargs.pop("path_to_data")
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop("path_to_vectors")
        emb_dim = kwargs.pop("emb_dim")
        voc_size = kwargs.pop("voc_size")
        min_freq = kwargs.pop("min_freq")
        fix_len = kwargs.pop("fix_len")
        min_len = kwargs.pop("min_len")
        train_dataset, test_dataset, valid_dataset = WikiText2(
            root=path_to_data,
            tokenizer=tokenizer,
            path_to_vectors=path_to_vectors,
            emb_dim=emb_dim,
            voc_size=voc_size,
            min_freq=min_freq,
            fix_len=fix_len,
            min_len=min_len,
        )

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(
            train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
        )
        self._valid_iter = DataLoader(
            valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
        )
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
        path_to_data = kwargs.pop("path_to_data")
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop("path_to_vectors")
        emb_dim = kwargs.pop("emb_dim")
        voc_size = kwargs.pop("voc_size")
        min_freq = kwargs.pop("min_freq")
        fix_len = kwargs.pop("fix_len")
        min_len = kwargs.pop("min_len")

        train_dataset, test_dataset, valid_dataset = WikiText103(
            root=path_to_data,
            tokenizer=tokenizer,
            path_to_vectors=path_to_vectors,
            emb_dim=emb_dim,
            voc_size=voc_size,
            min_freq=min_freq,
            fix_len=fix_len,
            min_len=min_len,
        )
        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(
            train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
        )
        self._valid_iter = DataLoader(
            valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
        )
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
        path_to_data = kwargs.pop("path_to_data")
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop("path_to_vectors")
        emb_dim = kwargs.pop("emb_dim")
        voc_size = kwargs.pop("voc_size")
        min_freq = kwargs.pop("min_freq")
        fix_len = kwargs.pop("fix_len")
        min_len = kwargs.pop("min_len")

        train_dataset, test_dataset, valid_dataset = YelpReviewPolarity(
            root=path_to_data,
            tokenizer=tokenizer,
            path_to_vectors=path_to_vectors,
            emb_dim=emb_dim,
            voc_size=voc_size,
            min_freq=min_freq,
            fix_len=fix_len,
            min_len=min_len,
            supervised_proportion=supervised_proportion,
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(
            train_dataset.data["weights"], num_samples=len(train_dataset.data["weights"])
        )
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            raise NotImplementedError("Distributed training and semi-supervised data loaders not implemented!")

        self._train_iter = DataLoader(
            train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
        )
        self._valid_iter = DataLoader(
            valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
        )
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

        dataset = eval(kwargs.pop("dataset"))
        path_to_data = kwargs.pop("path_to_data")
        super().__init__(device, rank, world_size, **kwargs)
        path_to_vectors = kwargs.pop("path_to_vectors")
        emb_dim = kwargs.pop("emb_dim")
        voc_size = kwargs.pop("voc_size")
        min_freq = kwargs.pop("min_freq")
        fix_len = kwargs.pop("fix_len")
        min_len = kwargs.pop("min_len")

        train_dataset, test_dataset, valid_dataset = dataset(
            root=path_to_data,
            tokenizer=tokenizer,
            path_to_vectors=path_to_vectors,
            emb_dim=emb_dim,
            voc_size=voc_size,
            min_freq=min_freq,
            fix_len=fix_len,
            min_len=min_len,
            supervised_proportion=supervised_proportion,
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(
            train_dataset.data["weights"], num_samples=len(train_dataset.data["weights"])
        )
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            raise NotImplementedError("Distributed training and semi-supervised data loaders not implemented!")

        self._train_iter = DataLoader(
            train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
        )
        self._valid_iter = DataLoader(
            valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
        )
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


class DataLoaderPretrained(ADataLoader):
    """
    Data loader for PTB with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):

        super().__init__(device, rank, world_size, **kwargs)

        path_to_data = kwargs.pop("path_to_data")
        self.path_to_pretrained_models = kwargs.pop("path_to_pretrained_models", path_to_data)
        min_len = kwargs.pop("min_len", 0)
        self._fix_len = kwargs.pop("fix_len", 256)

        pretrained_tokenizer = kwargs.pop("pretrained_tokenizer", None)
        assert pretrained_tokenizer is not None, "no pretrained tokenizer specified"

        train_dataset, test_dataset, valid_dataset = self.get_datasets(path_to_data, pretrained_tokenizer, min_len)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(
            train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
        )
        self._valid_iter = DataLoader(
            valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
        )
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)

        self._pad_token_id = train_dataset.get_pad_token_id()

        self._num_added_tokens = train_dataset.get_num_added_tokens()
        self._tokenizer = train_dataset.tokenizer_list[-1]

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
    def tokenizer(self):
        return self._tokenizer

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
    def vocab(self):  # for compatibility with TextTrainer
        return None


class DataLoaderPTBPretrained(DataLoaderPretrained):
    """
    Data loader for PTB with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):

        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, pretrained_tokenizer, min_len):
        return PennTreebankPretrained(
            root=path_to_data,
            pretrained_tokenizer=pretrained_tokenizer,
            fix_len=self._fix_len,
            min_len=min_len,
            path_to_pretrained_models=self.path_to_pretrained_models,
        )


class DataLoaderYahooPretrained(DataLoaderPretrained):
    """
    Data loader for YahooAnswers with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, pretrained_tokenizer, min_len):
        return YahooAnswersPretrained(
            root=path_to_data,
            pretrained_tokenizer=pretrained_tokenizer,
            fix_len=self._fix_len,
            min_len=min_len,
            path_to_pretrained_models=self.path_to_pretrained_models,
        )


class DataLoaderYelpPretrained(DataLoaderPretrained):
    """
    Data loader for YahooAnswers with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, pretrained_tokenizer, min_len):
        return YelpReviewPretrained(
            root=path_to_data,
            pretrained_tokenizer=pretrained_tokenizer,
            fix_len=self._fix_len,
            min_len=min_len,
            path_to_pretrained_models=self.path_to_pretrained_models,
        )


class DataLoaderWiki103Pretrained(DataLoaderPretrained):
    """
    Data loader for Wiki103 with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, pretrained_tokenizer, min_len):
        return WikiText103Pretrained(
            root=path_to_data,
            pretrained_tokenizer=pretrained_tokenizer,
            fix_len=self._fix_len,
            min_len=min_len,
            path_to_pretrained_models=self.path_to_pretrained_models,
        )


class DataLoaderWiki2Pretrained(DataLoaderPretrained):
    """
    Data loader for Wiki2 with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, pretrained_tokenizer, min_len):
        return WikiText2Pretrained(
            root=path_to_data,
            pretrained_tokenizer=pretrained_tokenizer,
            fix_len=self._fix_len,
            min_len=min_len,
            path_to_pretrained_models=self.path_to_pretrained_models,
        )


class DataLoaderWikiOptimusPretrained(DataLoaderPretrained):
    """
    Data loader for Wiki2 with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, pretrained_tokenizer, min_len):
        return WikiOptimusPretrained(
            root=path_to_data, fix_len=self._fix_len, min_len=min_len, path_to_pretrained_models=self.path_to_pretrained_models
        )


class DataLoaderAtomic(ADataLoader):
    """
    Data loader for ATOMIC with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        path_to_data = kwargs.pop("path_to_data")
        self.path_to_pretrained_models = kwargs.pop("path_to_pretrained_models", path_to_data)

        super().__init__(device, rank, world_size, **kwargs)
        min_len = kwargs.pop("min_len", 1)
        self._fix_len = kwargs.pop("fix_len", 64)
        self.add_gen_token = kwargs.pop("add_gen_token", False)
        path_to_posterior_samples = kwargs.pop("path_to_posterior_samples", None)
        if path_to_posterior_samples is None:
            train_dataset, test_dataset, valid_dataset, test_unshuffled_unique = self.get_datasets(
                path_to_data, min_len, self.add_gen_token)
        else:
            train_dataset, test_dataset, valid_dataset, test_unshuffled_unique = self.get_datasets(
                path_to_data, min_len, self.add_gen_token, path_to_posterior_samples
            )
        get_test_unshuffled = kwargs.pop("get_test_unshuffled", False)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        if get_test_unshuffled:
            self._train_iter = DataLoader(train_dataset, drop_last=False, **kwargs)
            self._valid_iter = DataLoader(valid_dataset, drop_last=False, **kwargs)
            self._test_iter = DataLoader(test_dataset, drop_last=False, **kwargs)
            self._test_sub_rel_unique = DataLoader(test_unshuffled_unique, drop_last=False, **kwargs)
        else:
            self._train_iter = DataLoader(
                train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs
            )
            self._valid_iter = DataLoader(
                valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs
            )
            self._test_iter = DataLoader(
                test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs
            )

        self._pad_token_id = train_dataset.get_pad_token_id()
        self._unk_token_id = train_dataset.get_unk_token_id()
        self._gen_token_id = train_dataset.get_gen_token_id() if self.add_gen_token else None

        self._num_added_tokens = train_dataset.get_num_added_tokens()
        self._num_relations = train_dataset.num_relations
        self._tokenizer = train_dataset.tokenizer

    @property
    def test_unshuffled(self):
        return self._test_unshuffled

    @property
    def test_sub_rel_unique(self):
        return self._test_sub_rel_unique

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
    def tokenizer(self):
        return self._tokenizer

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def unk_token_id(self):
        return self._unk_token_id

    @property
    def gen_token_id(self):
        return self._unk_token_id

    @property
    def fix_len(self):
        return self._fix_len

    @property
    def num_added_tokens(self):
        return self._num_added_tokens

    @property
    def num_relations(self):
        return self._num_relations

    @property
    def vocab(self):  # for compatibility with TextTrainer
        return None


class DataLoaderAtomic2(DataLoaderAtomic):
    """
    Data loader for ATOMIC with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, min_len, add_gen_token):
        return Atomic2(root=path_to_data, fix_len=self.fix_len, min_len=min_len, add_gen_token=add_gen_token)


class DataLoaderAtomic2FinetunedKL(DataLoaderAtomic):
    """
    Data loader for ATOMIC with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        assert not kwargs.get("path_to_posterior_samples", None) is None
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, min_len, add_gen_token, path_to_posterior_samples):
        return Atomic2(
            root=path_to_data,
            fix_len=self.fix_len,
            min_len=min_len,
            add_gen_token=add_gen_token,
            path_to_posterior_samples=path_to_posterior_samples,
        )


class DataLoaderAtomic2020(DataLoaderAtomic):
    """
    Data loader for ATOMIC with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, min_len, add_gen_token):
        return Atomic2020(root=path_to_data, fix_len=self.fix_len, min_len=min_len, add_gen_token=add_gen_token)

class DataLoaderConceptNetPretrained(DataLoaderAtomic):
    """
    Data loader for ConceptNet with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, min_len, add_gen_token):
        return ConceptNet(root=path_to_data, fix_len=self.fix_len, min_len=min_len, add_gen_token=add_gen_token)

class DataLoaderConceptNet5(DataLoaderAtomic):
    """
    Data loader for ConceptNet with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data, min_len, add_gen_token):
        return ConceptNet5(root=path_to_data, fix_len=self.fix_len, min_len=min_len, add_gen_token=add_gen_token)
