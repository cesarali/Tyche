import io
import logging
import os

from typing import Dict, List

import numpy as np
import torch
import random
from torchtext.data import get_tokenizer, numericalize_tokens_from_iterator

from torchtext.utils import download_from_url
from torchtext.vocab import Vocab
from tqdm import tqdm
from tyche.data.experimental.datasets.language_modeling import _get_datafile_path
from tyche.utils.helper import get_file_line_number

from .vocab import build_vocab_from_iterator

UNK = 0
PAD = 1
SOS = 2
EOS = 3

random.seed(1)


def numericalize(vocab, iterator):
    for label, tokens in iterator:
        yield int(label[0]), iter(vocab[token] for token in tokens)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull
    """

    def __init__(self, data: Dict[str, List], vocab):
        """Initiate text-classification dataset.

        Arguments:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(TextClassificationDataset, self).__init__()
        self.__n_supervised = data.pop('n_supervised', None)
        self.__n_unsupervised = data.pop('n_unsupervised', None)
        self.data = data
        self.vocab = vocab
        self.PAD = '<pad>'
        self.SOS = '<sos>'
        self.EOS = '<eos>'

        self.__get_item = self.__get_item_supervised if self.__n_supervised is None else self.__get_item_semisupervised

    def __getitem__(self, i):
        return self.__get_item(i)

    def __get_item_supervised(self, i):
        return {'input': np.asarray(self.data['input'][i], dtype=np.int64),
                'target': np.asarray(self.data['target'][i], dtype=np.int64),
                'length': np.asarray(self.data['length'][i]),
                'label': np.asarray(self.data['labels'][i])}

    def __get_item_semisupervised(self, i):
        return {'input': np.asarray(self.data['input'][i], dtype=np.int64),
                'target': np.asarray(self.data['target'][i], dtype=np.int64),
                'length': np.asarray(self.data['length'][i]),
                'label': np.asarray(self.data['labels'][i]),
                'supervised': np.asarray(self.data['supervised'][i])}

    def __len__(self):
        return len(self.data['input'])

    @property
    def n_supervised(self):
        return self.__n_supervised

    @property
    def n_unsupervised(self):
        return self.__n_unsupervised

    @property
    def vocab(self):
        return self.__vocab

    @vocab.setter
    def vocab(self, v):
        self.__vocab = v

    def reverse(self, batch):

        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.EOS) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.SOS, self.PAD)

        batch = [filter(filter_special, ex) for ex in batch]

        return [' '.join(ex) for ex in batch]


# adds labels at the beginning of lines for YahooAnswers
def preprocess_yahoo(file):
    src = open(file, "rt")
    label = -1
    file_iter = [row for row in src]
    data = ''
    for i, line in enumerate(file_iter):
        if i % (len(file_iter) / 10) == 0:
            label += 1
        data += str(label) + '\t' + line
    src.close()
    dest = open(file, "wt")
    dest.write(data)
    dest.close()
    os.rename(file, file.replace('val', 'valid'))


def _setup_datasets(
        dataset_name, emb_dim, voc_size, fix_len, min_len=0, path_to_vectors=None, min_freq=1,
        root="data",
        supervised_proportion: float = 0.9,
        vocab=None,
        tokenizer=None,
        data_select=("train", "test", "valid"),
):
    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset({"train", "test", "valid"}):
        raise TypeError("Given data selection {} is not supported!".format(data_select))
    if dataset_name in ['YelpReviewPolarity', 'YahooAnswers']:
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        for key in data_select:
            url_ = URLS[dataset_name][select_to_index[key]]
            _, filename = os.path.split(url_)
            path_ = os.path.join(root, filename)
            if dataset_name == 'YahooAnswers':
                path_ = path_.replace('val', 'valid')
            if os.path.exists(path_):
                extracted_files.append(path_)
            else:
                file = download_from_url(url_, root=root)
                if dataset_name == 'YahooAnswers':
                    preprocess_yahoo(file)
                    file = file.replace('val', 'valid')
                extracted_files.append(file)

    # Cache raw text iterable dataset
    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)

    if vocab is None:
        if 'train' not in _path.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on {}'.format(_path['train']))

        with io.open(_path['train'], encoding="utf8") as f:
            txt_iter = iter(tokenizer(row.split('\t')[1]) for row in f)
            vocab = build_vocab_from_iterator(txt_iter, min_freq=min_freq, voc_size=voc_size, emb_dim=emb_dim, path_to_vectors=path_to_vectors)
            logging.info('Vocab has {} entries'.format(len(vocab)))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    data = dict()

    for item in _path.keys():
        logging.info('Creating {} data'.format(item))
        f = io.open(_path[item], encoding="utf8")
        tokenize = lambda sentiment, text: (sentiment, tokenizer(text))
        txt_iter = iter(tokenize(*row.split('\t')) for row in f)
        _iter = numericalize(vocab, txt_iter)
        idx = 0
        N = get_file_line_number(_path[item])
        unsupervised_proportion = (1. - supervised_proportion)
        U = int(unsupervised_proportion * N)
        S = N - U
        unsupervised = random.sample(range(N), U)
        unsupervised = dict.fromkeys(unsupervised)
        inputs, targets, lengths, labels, supervised, weights = [], [], [], [], [], []
        for label, tokens in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset', total=N):
            tokens = list(tokens)
            size = len(tokens)
            if size < min_len:
                continue

            tokens = tokens[:fix_len - 1]
            input_ = [SOS] + tokens
            target_ = tokens + [EOS]
            assert len(input_) == len(target_)
            size = len(input_)
            padding = [PAD] * (fix_len - size)
            inputs.append(input_ + padding)
            targets.append(target_ + padding)
            lengths.append(size)
            labels.append(label)
            if idx not in unsupervised:
                supervised.append(True)
                weights.append((supervised_proportion / S))
            else:
                supervised.append(False)
                weights.append(unsupervised_proportion / U)

            idx += 1
        data[item] = {'input': inputs, 'target': targets, 'length': lengths,
                      'labels': labels}
        if item == 'train':
            data[item].update({'supervised': supervised,
                               'n_supervised': S,
                               'n_unsupervised': U,
                               'weights': weights})
        f.close()
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))
    return tuple(TextClassificationDataset(data[d], vocab) for d in data_select)


def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create text classification dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from tyche.data.experimental.datasets import YelpReviewPolarity
        >>> from nltk.tokenize import TweetTokenizer
        >>> train, test, valid = YelpReviewPolarity(ngrams=3)
        >>> tokenizer = TweetTokenizer(preserve_case=False).tokenize
        >>> train, test, valid = YelpReviewPolarity(tokenizer=tokenizer)
        >>> train, = YelpReviewPolarity(tokenizer=tokenizer, data_select='train')

    """
    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create text classification dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from tyche.data.experimental.datasets import YelpReviewPolarity
        >>> from nltk.tokenize import TweetTokenizer
        >>> train, test, valid = YelpReviewPolarity(ngrams=3)
        >>> tokenizer = TweetTokenizer(preserve_case=False).tokenize
        >>> train, test, valid = YelpReviewPolarity(tokenizer=tokenizer)
        >>> train, = YelpReviewPolarity(tokenizer=tokenizer, data_select='train')

    """
    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create text classification dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well. A custom tokenizer is callable
            function with input of a string and output of a token list.
        data_select: a string or tuple for the returned datasets
            (Default: ('train', 'test'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from tyche.data.experimental.datasets import YelpReviewPolarity
        >>> from nltk.tokenize import TweetTokenizer
        >>> train, test, valid = YelpReviewPolarity(ngrams=3)
        >>> tokenizer = TweetTokenizer(preserve_case=False).tokenize
        >>> train, test, valid = YelpReviewPolarity(tokenizer=tokenizer)
        >>> train, = YelpReviewPolarity(tokenizer=tokenizer, data_select='train')

    """
    kwargs['root'] += '/yahoo'

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'YelpReviewPolarity':
        ['https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.train.txt',
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.test.txt',
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.valid.txt'],
    'YahooAnswers':
        ['https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/train.txt',
         'https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/test.txt',
         'https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/val.txt']
}

DATASETS = {

    "YelpReviewPolarity": YelpReviewPolarity,
    "YahooAnswers": YahooAnswers

}

LABELS = {
    "AG_NEWS": {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"},
    "SogouNews": {
        1: "Sports",
        2: "Finance",
        3: "Entertainment",
        4: "Automobile",
        5: "Technology",
    },
    "DBpedia": {
        1: "Company",
        2: "EducationalInstitution",
        3: "Artist",
        4: "Athlete",
        5: "OfficeHolder",
        6: "MeanOfTransportation",
        7: "Building",
        8: "NaturalPlace",
        9: "Village",
        10: "Animal",
        11: "Plant",
        12: "Album",
        13: "Film",
        14: "WrittenWork",
    },
    "YelpReviewPolarity": {1: "Negative polarity", 2: "Positive polarity"},
    "YelpReviewFull": {
        1: "score 1",
        2: "score 2",
        3: "score 3",
        4: "score 4",
        5: "score 5",
    },
    "YahooAnswers": {
        1: "Society & Culture",
        2: "Science & Mathematics",
        3: "Health",
        4: "Education & Reference",
        5: "Computers & Internet",
        6: "Sports",
        7: "Business & Finance",
        8: "Entertainment & Music",
        9: "Family & Relationships",
        10: "Politics & Government",
    },
    "AmazonReviewPolarity": {1: "Negative polarity", 2: "Positive polarity"},
    "AmazonReviewFull": {
        1: "score 1",
        2: "score 2",
        3: "score 3",
        4: "score 4",
        5: "score 5",
    },
    "IMDB": {0: "Negative", 1: "Positive"}}
