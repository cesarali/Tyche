import io
import logging
from collections import Counter

import torch
from torchtext.data.functional import numericalize_tokens_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab
from tqdm import tqdm

URLS = {
    'WikiText2':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt']
}

PAD = 1
SOS = 2
EOS = 3


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank

    """

    def __init__(self, data, vocab):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)

        """

        super(LanguageModelingDataset, self).__init__()
        self.data = data
        self.vocab = vocab

        self.PAD = '<pad>'
        self.SOS = '<sos>'
        self.EOS = '<eos>'

    def __getitem__(self, i):
        return self.data[0][i], self.data[1][i]

    def __len__(self):
        return len(self.data[0])

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.data[0][i], self.data[1][i]

    def get_vocab(self):
        return self.vocab

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


def _get_datafile_path(key, extracted_files):
    for fname in extracted_files:
        if key in fname:
            return fname


def build_vocab_from_iterator(iterator, emb_dim, voc_size, min_freq, path_to_vectors):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
    """

    counter = Counter()
    with tqdm(unit='lines', desc='Building Vocabulary') as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = Vocab(counter, max_size=voc_size, min_freq=min_freq, vectors=emb_dim, vectors_cache=path_to_vectors,
                       specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    return word_vocab


def _setup_datasets(dataset_name, emb_dim, voc_size, fix_len, path_to_vectors=None, min_freq=1,
                    tokenizer=get_tokenizer("basic_english"),
                    root='.data', vocab=None, removed_tokens=[],
                    data_select=('train', 'test', 'valid'), ):
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')

    if dataset_name == 'PennTreebank':
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        extracted_files = [download_from_url(URLS['PennTreebank'][select_to_index[key]],
                                             root=root) for key in data_select]
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)

    if vocab is None:
        if 'train' not in _path.keys():
            raise TypeError("Must pass a vocab if train is not selected.")
        logging.info('Building Vocab based on {}'.format(_path['train']))
        txt_iter = iter(tokenizer(row) for row in io.open(_path['train'],
                                                          encoding="utf8"))
        vocab = build_vocab_from_iterator(txt_iter, min_freq=min_freq, voc_size=voc_size, emb_dim=emb_dim, path_to_vectors=path_to_vectors)
        logging.info('Vocab has {} entries'.format(len(vocab)))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    data = {}

    for item in _path.keys():
        tmp_tokens = []
        tmp_len = []
        logging.info('Creating {} data'.format(item))
        txt_iter = iter(tokenizer(row) for row in io.open(_path[item],
                                                          encoding="utf8"))
        _iter = numericalize_tokens_from_iterator(vocab, txt_iter, removed_tokens)
        for tokens in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset'):
            tokens_ = [token_id for token_id in tokens]
            size = len(tokens_)
            if size == 0 or tokens_.count(vocab['=']) >= 2:
                continue

            if size > fix_len - 2:
                tokens_ = [SOS] + tokens_[:fix_len - 2] + [EOS]
                size = fix_len
            else:
                tokens_ = [SOS] + tokens_ + [EOS] + [PAD] * (fix_len - size - 2)

            tmp_tokens.append(tokens_)
            tmp_len.append(size)
        data[item] = (tmp_tokens, tmp_len)
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))

    return tuple(LanguageModelingDataset((torch.tensor(data[d][0]).long(), torch.tensor(data[d][1])), vocab)
                 for d in data_select)


def WikiText2(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText2(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText103(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: [])
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.

    Examples:
        >>> from torchtext.experimental.datasets import PennTreebank
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = PennTreebank(tokenizer=tokenizer, vocab=vocab,
                                          data_select='valid')

    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)