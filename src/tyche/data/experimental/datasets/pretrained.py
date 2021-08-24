import io
import logging
import os
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from tyche.data.experimental.datasets.language_modeling import LanguageModelingDataset, _get_datafile_path
import torch
from transformers import GPT2TokenizerFast


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


class LanguageModelingDatasetPretrained(LanguageModelingDataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank

    """

    def __init__(self, data, tokenizer):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.
        """

        super(LanguageModelingDatasetPretrained, self).__init__(data, tokenizer)
        self.data = data
        self.tokenizer = tokenizer

        self.PAD = tokenizer.eos_token
        self.SOS = tokenizer.bos_token
        self.EOS = tokenizer.eos_token

    def __getitem__(self, i):
        return {'input': np.asarray(self.data[i]['input'], dtype=np.int64),
                'target': np.asarray(self.data[i]['target'], dtype=np.int64),
                'length': np.asarray(self.data[i]['length']),
                'attn_mask': np.asarray(self.data[i]['attn_mask'])}

    def __iter__(self):
        for i in range(self.__len__()):
            yield {'input': np.asarray(self.data[i]['input']),
                   'target': np.asarray(self.data[i]['target']),
                   'length': np.asarray(self.data[i]['length']),
                   'attn_mask': np.asarray(self.data[i]['attn_mask'])}


    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def reverse(self, batch):

        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [self.tokenizer.convert_ids_to_tokens(ex) for ex in batch]  # denumericalize
        print(batch)

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex[1:], self.EOS) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.SOS, self.PAD)

        batch = [filter(filter_special, ex) for ex in batch]

        return [' '.join(ex) for ex in batch]


def _setup_datasets(dataset_name, fix_len, min_len=0, min_freq=1,
                    pretrained_model='gpt2',
                    root='./data', removed_tokens=[],
                    data_select=('train', 'test', 'valid'), ):
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset({'train', 'test', 'valid'}):
        raise TypeError('data_select is not supported!')

    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model)

    if dataset_name == 'PennTreebankPretrained':
        extracted_files = []

        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        for key in data_select:
            url_ = URLS['PennTreebank'][select_to_index[key]]
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

    SOS = tokenizer.bos_token_id
    EOS = tokenizer.eos_token_id
    PAD = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    for item in _path.keys():
        data_set = defaultdict(dict)
        logging.info('Creating {} data'.format(item))
        _iter = iter(row for row in io.open(_path[item], encoding="utf8"))
        id = 0
        for row in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset'):
            row = row[:-2]  # remove white space and \n in the end of each line
            tokens_attns = tokenizer(row)
            tokens = tokens_attns['input_ids']
            tokens_ = [token_id for token_id in tokens]
            size = len(tokens_)
            if size < min_len or tokens_.count(vocab.get('=', -1)) >= 2:
                continue

            tokens_ = [SOS] + tokens_[:fix_len - 1] + [EOS]
            input_ = tokens_[:-1]
            target_ = tokens_[1:]
            assert len(input_) == len(target_)
            size = len(input_)
            data_set[id]['input'] = input_ + [PAD] * (fix_len - size)
            data_set[id]['target'] = target_ + [PAD] * (fix_len - size)
            data_set[id]['length'] = size
            data_set[id]['attn_mask'] = tokens_attns['attention_mask'] + [1] + [0] * (fix_len - size)

            id += 1
        data[item] = data_set
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))

    return tuple(LanguageModelingDatasetPretrained(data[d], tokenizer) for d in data_select)


def PennTreebankPretrained(*args, **kwargs):

    return _setup_datasets(*(("PennTreebankPretrained",) + args), **kwargs)