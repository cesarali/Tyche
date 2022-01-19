import io
import logging
import os
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from tyche.data.experimental.datasets.language_modeling import LanguageModelingDataset, _get_datafile_path
import torch
from torchtext.utils import download_from_url, extract_archive
from transformers import GPT2Tokenizer, BertTokenizer
import regex as re


URLS = {
    'WikiText2Pretrained':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'WikiText103Pretrained':
        'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'PennTreebankPretrained':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'YahooAnswersPretrained':
        ['https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/train.txt',
         'https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/test.txt',
         'https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/val.txt'],
    'YelpReviewPretrained':
        ['https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.train.txt',
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.test.txt',
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.valid.txt'],
    'WikiOptimusPretrained':
        'https://textae.blob.core.windows.net/optimus/data/datasets/wikipedia_json_64_filtered.zip'

}


class LanguageModelingDatasetPretrained(LanguageModelingDataset):
    """Defines a dataset for language modeling using pretrained tokenizers from huggingface transformers.
       Currently, we only support the following datasets:

             - WikiText103
             - PennTreebank
             - YahooAnswers

        minibatches are dictionaries with the following content:
            for t_id, tokenizer in enumerate(tokenizer_list):
                'input_{t_id}': text tokenized by tokenizer with SOS at beginning
                'target_{t_id}': text tokenized by tokenizer with EOS at the end
                'length_{t_id}': number of tokens when tokenized by tokenizer
                'attn_mask_{t_id}': attention mask for transformers
            'label': label of the text in case of YahooAnswers

    """

    def __init__(self, data, tokenizer_list, num_added_tokens):
        """Initiate language modeling dataset using pretrained tokenizers from huggingface.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            tokenizer_list: list of huggingface tokenizers
            num_added_tokens: number of tokens that were newly added to the vocabulary of each tokenizer
        """

        super(LanguageModelingDatasetPretrained, self).__init__(data, tokenizer_list)
        self.data = data
        self.tokenizer_list = tokenizer_list
        self.tokenizer = tokenizer_list[-1]

    def __getitem__(self, i):
        minibatch = dict()
        for tok_id in range(len(self.tokenizer_list)):
            minibatch.update({'input_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['input'], dtype=np.int64),
                              'attn_mask_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['attn_mask']),
                              'length_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['length'])})
            if 'target' in self.data[i][tok_id].keys():
                minibatch.update({'target_{}'.format(tok_id): np.asarray(self.data[i][tok_id]['target'], dtype=np.int64)})
        if 'label' in self.data[i][0].keys():
            minibatch.update({'label': np.asarray(self.data[i][0]['label'])})
        return minibatch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    @staticmethod
    def get_pad_token_id():
        return -100

    @staticmethod
    def get_num_added_tokens():
        return 0

    def reverse(self, batch):
        batch[batch == -100] = self.tokenizer.eos_token_id
        sentences = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
        return sentences

# add label at beginning of each line of yahoo
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


# remove empty lines and section headers from wiki103
def preprocess_wiki103(file):
    src = open(file, "rt")
    file_iter = [row for row in src]
    data = ''
    for i, line in enumerate(file_iter):
        if len(line) <= 2:
            continue
        if line.count(' = ') >= 2:
            continue
        data += line
    src.close()
    dest = open(file, "wt")
    dest.write(data)
    dest.close()


def _setup_datasets(dataset_name,
                    fix_len,
                    min_len=0,
                    min_freq=1,
                    path_to_pretrained_models='./data',
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
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                                          cache_dir=path_to_pretrained_models)
        elif model_name == 'BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                          cache_dir=path_to_pretrained_models)
        else:
            raise ValueError('pretrained tokenizer {} not supported! Choose one of [GPT2, BERT]'.format(model_name))

        tokenizer_list.append(tokenizer)
        # for compatibility with the data loaders
        special_tokens = []

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

    elif dataset_name in ['YahooAnswersPretrained', 'YelpReviewPretrained']:
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        for key in data_select:
            url_ = URLS[dataset_name][select_to_index[key]]
            _, filename = os.path.split(url_)
            path_ = os.path.join(root, filename)
            if not os.path.exists(path_.replace('val', 'valid')):
                path_ = download_from_url(url_, root=root)
                if dataset_name == 'YahooAnswersPretrained':
                    preprocess_yahoo(path_)
                os.rename(path_, path_.replace('val', 'valid'))
            extracted_files.append(path_.replace('val', 'valid'))

    elif dataset_name == 'WikiText103Pretrained':
        url_ = URLS[dataset_name]
        _, filename = os.path.split(url_)
        dataset_tar = os.path.join(root, filename)
        if not os.path.exists(dataset_tar):
            dataset_tar = download_from_url(url_, root=root)
        extracted_files = extract_archive(dataset_tar)
        for file in extracted_files:
            preprocess_wiki103(file)

    elif dataset_name == 'WikiText2Pretrained':
        url_ = URLS[dataset_name]
        _, filename = os.path.split(url_)
        dataset_tar = os.path.join(root, filename)
        if not os.path.exists(dataset_tar):
            dataset_tar = download_from_url(url_, root=root)
        extracted_files = extract_archive(dataset_tar)
        for file in extracted_files:
            print(file)
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

    for item in _path.keys():
        data_set = defaultdict(lambda: defaultdict(dict))
        logging.info('Creating {} data'.format(item))
        _iter = iter(row for row in io.open(_path[item], encoding="utf8"))
        id = 0
        for row in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset'):
            row = row[:-1]  # remove \n at the end of each line

            ### data set specific alterations ###
            if dataset_name == 'WikiText2Pretrained' and (row == '' or row[0] == '='):
                continue
            if dataset_name in ['YahooAnswersPretrained', 'YelpReviewPretrained']:
                data_set[id][0]['label'] = int(row[0])
                row = row[2:]  # remove the label (and the space after it)

            for t_id, tokenizer in enumerate(tokenizer_list):

                if pretrained_tokenizer[t_id] == 'BERT':
                    ### BERT tokenizer ###

                    tokens_attns = tokenizer(row,
                                             truncation=True,
                                             max_length=fix_len,
                                             return_length=True,
                                             add_special_tokens=True)

                    if tokens_attns['length'] < min_len:
                        continue

                    pad_len = fix_len - tokens_attns['length']

                    data_set[id][t_id]['length'] = tokens_attns['length']
                    data_set[id][t_id]['input'] = tokens_attns['input_ids'] + [tokenizer.pad_token_id] * pad_len
                    data_set[id][t_id]['attn_mask'] = tokens_attns['attention_mask'] + [0] * pad_len
                else:
                    ### GPT2 tokenizer ###
                    tokens_attns = tokenizer(row,
                                             truncation=True,
                                             max_length=fix_len-1,
                                             return_length=True)


                    pad_len = fix_len - tokens_attns['length'] - 1

                    data_set[id][t_id]['length'] = tokens_attns['length']
                    data_set[id][t_id]['input'] = [tokenizer.bos_token_id] + tokens_attns['input_ids'] + \
                                                  [tokenizer.eos_token_id] * pad_len
                    data_set[id][t_id]['target'] = tokens_attns['input_ids'] + [tokenizer.eos_token_id] + \
                                                   [-100] * pad_len
                    data_set[id][t_id]['attn_mask'] = tokens_attns['attention_mask'] + [1] + [0] * pad_len

                    assert len(data_set[id][t_id]['input']) == len(data_set[id][t_id]['target']) \
                           == len(data_set[id][t_id]['attn_mask']) == fix_len

            id += 1
        data[item] = data_set
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))


    return tuple(LanguageModelingDatasetPretrained(data[d], tokenizer_list, len(special_tokens)) for d in data_select)

def PennTreebankPretrained(*args, **kwargs):
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


def YahooAnswersPretrained(*args, **kwargs):
    return _setup_datasets(*(("YahooAnswersPretrained",) + args), **kwargs)

def YelpReviewPretrained(*args, **kwargs):
    return _setup_datasets(*(("YelpReviewPretrained",) + args), **kwargs)

def WikiText103Pretrained(*args, **kwargs):
    return _setup_datasets(*(("WikiText103Pretrained",) + args), **kwargs)

def WikiText2Pretrained(*args, **kwargs):
    return _setup_datasets(*(("WikiText2Pretrained",) + args), **kwargs)
