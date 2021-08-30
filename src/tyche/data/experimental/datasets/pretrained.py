import io
import logging
import os
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from tyche.data.experimental.datasets.language_modeling import LanguageModelingDataset, _get_datafile_path
import torch
from torchtext.utils import download_from_url, extract_archive
from tyche.data.experimental.datasets.my_tokenization_gpt2_fast import GPT2TokenizerLowerCase
from transformers import GPT2TokenizerFast, BertTokenizerFast
import regex as re


URLS = {
    'WikiText2Pretrained':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103Pretrained':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebankPretrained':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt']
}


class LanguageModelingDatasetPretrained(LanguageModelingDataset):
    """Defines a dataset for language modeling using pretrained tokenizers from huggingface transformers.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank

    """

    def __init__(self, data, tokenizer_list):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.
        """

        super(LanguageModelingDatasetPretrained, self).__init__(data, tokenizer_list)
        self.data = data
        self.tokenizer_list = tokenizer_list

        self.PAD = tokenizer_list[0].pad_token
        self.SOS = tokenizer_list[0].bos_token
        self.EOS = tokenizer_list[0].eos_token

    def __getitem__(self, i):
        minibatch = dict()
        for tok_id in range(len(self.tokenizer_list)):
            minibatch.update({'input_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['input'], dtype=np.int64),
                               'target_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['target'], dtype=np.int64),
                               'length_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['length']),
                               'attn_mask_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['attn_mask'])})
        return minibatch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]


    def get_pad_token_id(self):
        return self.tokenizer_list[0].pad_token_id

    def reverse(self, batch):

        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [self.tokenizer_list[-1].convert_ids_to_tokens(ex) for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex[1:], self.EOS) for ex in batch]  # trim past first eos

        def filter_special(tok):
            return tok not in (self.SOS, self.PAD)

        batch = [filter(filter_special, ex) for ex in batch]

        return [' '.join(ex) for ex in batch]


def _setup_datasets(dataset_name, fix_len, min_len=0, min_freq=1,
                    pretrained_tokenizer=['GPT2', 'GPT2'],
                    root='./data',
                    data_select=('train', 'test', 'valid'), ):
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset({'train', 'test', 'valid'}):
        raise TypeError('data_select is not supported!')

    # get the pretrained tokenizers
    tokenizer_list = []
    for model_name in pretrained_tokenizer:
        if model_name == 'GPT2':
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif model_name == 'BERT':
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            raise ValueError('pretrained tokenizer {} not supported! Choose one of [GPT2, BERT]'.format(model_name))
        tokenizer_list.append(tokenizer)



    for tokenizer in tokenizer_list:
        tokenizer.add_special_tokens({'unk_token': '<unk>',
                                      'pad_token': '<pad>',
                                      'bos_token': '<bos>',
                                      'eos_token': '<eos>'})

    if dataset_name == 'PennTreebankPretrained':
        extracted_files = []

        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        for key in data_select:
            url_ = URLS['PennTreebankPretrained'][select_to_index[key]]
            _, filename = os.path.split(url_)
            path_ = os.path.join(root, filename)
            if os.path.exists(path_):
                extracted_files.append(path_)
            else:
                extracted_files.append(download_from_url(url_, root=root))

    elif dataset_name in URLS:
        url_ = URLS[dataset_name]
        _, filename = os.path.split(url_)
        dataset_tar = os.path.join(root, filename)
        if not os.path.exists(dataset_tar):
            dataset_tar = download_from_url(url_, root=root)
        extracted_files = extract_archive(dataset_tar)
    else:
        extracted_files = []
        for key in data_select:

            file_ = os.path.join(root, f'{key}.txt')
            if not os.path.exists(file_):
                raise FileExistsError(f'File cannot be found at location {file_}')
            extracted_files.append(file_)

    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)

    data = dict()

    SOS = tokenizer_list[0].bos_token_id
    EOS = tokenizer_list[0].eos_token_id
    PAD = tokenizer_list[0].pad_token_id

    vocab_list = [tokenizer.get_vocab() for tokenizer in tokenizer_list]
    for item in _path.keys():
        data_set = defaultdict(lambda: defaultdict(dict))
        logging.info('Creating {} data'.format(item))
        _iter = iter(row for row in io.open(_path[item], encoding="utf8"))
        id = 0
        for row in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset'):
            row = row[:-2]  # remove white space and \n in the end of each line
            if dataset_name == 'WikiText2Pretrained' and (row == '' or row[0] == '='):
                continue
            for t_id, tokenizer in enumerate(tokenizer_list):
                tokens_attns = tokenizer(row)
                tokens = tokens_attns['input_ids']
                tokens_ = [token_id for token_id in tokens]
                size = len(tokens_)

                if size < min_len:
                    continue

                tokens_ = [SOS] + tokens_[:fix_len - 1] + [EOS]
                input_ = tokens_[:-1]
                target_ = tokens_[1:]
                attn_mask = tokens_attns['attention_mask'][:fix_len - 1]
                assert len(input_) == len(target_)
                size = len(input_)
                data_set[id][t_id]['input'] = input_ + [PAD] * (fix_len - size)
                data_set[id][t_id]['target'] = target_ + [PAD] * (fix_len - size)
                data_set[id][t_id]['length'] = size
                data_set[id][t_id]['attn_mask'] = attn_mask + [1] + [0] * (fix_len - size)
            id += 1
        data[item] = data_set
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))


    return tuple(LanguageModelingDatasetPretrained(data[d], tokenizer_list) for d in data_select)


def PennTreebankPretrained(*args, **kwargs):
    def PennTreebank(*args, **kwargs):
        """ Defines PennTreebank datasets.

        Create language modeling dataset: PennTreebank
        Separately returns the train/test/valid set

        Arguments:
            pretrained_tokenizer: list of strings from {'GPT2', 'BERT'} that correspond to the tokenizers of huggingface
                transformers {'gpt2', 'bert-base-uncased'}
            root: Directory where the datasets are saved. Default: ".data"
            data_select: a string or tupel for the returned datasets
                (Default: ('train', 'test','valid'))
                By default, all the three datasets (train, test, valid) are generated. Users
                could also choose any one or two of them, for example ('train', 'test') or
                just a string 'train'. If 'train' is not in the tuple or string, a vocab
                object should be provided which will be used to process valid and/or test
                data.

        The returned data sets contain a dictionary with keys
            'input_{i}'
            'target_{i}'
            'length_{i}'
            'attn_mask_{i}'
            for each tokenizer, where {i} is the index of each tokenizer in pretrained_tokenizer

        """

    return _setup_datasets(*(("PennTreebankPretrained",) + args), **kwargs)

def WikiText2Pretrained(*args, **kwargs):
    return _setup_datasets(*(("WikiText2Pretrained",) + args), **kwargs)
