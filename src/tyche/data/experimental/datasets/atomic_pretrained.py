import io
import logging
import os
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from tyche.data.experimental.datasets.language_modeling import LanguageModelingDataset, _get_datafile_path
import torch
from torchtext.utils import download_from_url, extract_archive
from transformers import GPT2TokenizerFast, BertTokenizerFast
import regex as re
import json

class AtomicDatasetPretrained(LanguageModelingDataset):
    """Defines a dataset for language modeling using pretrained tokenizers from huggingface transformers.
       Currently, we only support the following datasets:

             - Atomic2
             - Atomic2020

        minibatches are dictionaries with the following content:
            for t_id, tokenizer in enumerate(tokenizer_list):
                'input': SOS + subject + relation + object
                'target': subject + relation + object + EOS
                'relation': relation
                'length': number of tokens in target
                'attn_mask_in': attention mask for transformers
                ''

    """

    def __init__(self, data, tokenizer, num_added_tokens):
        """Initiate language modeling dataset using pretrained tokenizers from huggingface.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            tokenizer_list: list of huggingface tokenizers
            num_added_tokens: number of tokens that were newly added to the vocabulary of each tokenizer
        """

        super(AtomicDatasetPretrained, self).__init__(data, tokenizer)
        self.data = data
        self.tokenizer = tokenizer

        self.num_added_tokens = num_added_tokens

    def __getitem__(self, i):
        return {'input_enc': np.asarray(self.data[i]['input_enc'], dtype=np.int64),
                'input_dec': np.asarray(self.data[i]['input_dec'], dtype=np.int64),
                'target': np.asarray(self.data[i]['target'], dtype=np.int64),
                'length': np.asarray(self.data[i]['length']),
                'token_type_ids': np.asarray(self.data[i]['token_type_ids']),
                'attn_mask_enc': np.asarray(self.data[i]['attn_mask_enc']),
                'attn_mask_dec': np.asarray(self.data[i]['attn_mask_dec']),
                'mask_subject_relation': np.asarray(self.data[i]['mask_subject_relation']),
                'relation': np.asarray(self.data[i]['relation'])}

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_unk_token_id(self):
        return self.tokenizer.unk_token_id

    def get_num_added_tokens(self):
        return self.num_added_tokens

    def reverse(self, batch):

        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [self.tokenizer.convert_ids_to_tokens(ex) for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.tokenizer.eos_token_id) for ex in batch]  # trim past first eos

        def filter_special(tok):
            return tok not in (self.SOS, self.PAD)

        batch = list(filter(filter_special, ex) for ex in batch)

        return [' '.join(ex) for ex in batch]

def _setup_datasets(dataset_name,
                    fix_len,
                    min_len=0,
                    min_freq=1,
                    root='./data',
                    data_select=('train', 'test', 'valid'), ):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset({'train', 'test', 'valid'}):
        raise TypeError('data_select is not supported!')

    if dataset_name == 'Atomic2':
        extra_tokens = ["<oEffect>", "<oReact>", "<oWant>", "<xAttr>", "<xEffect>", "<xIntent>", "<xNeed>", "<xReact>", "<xWant>",
                        'none', '___', 'PersonX', 'PersonY']
        special_tokens = {'unk_token': '<unk>',
                          'pad_token': '<pad>',
                          'bos_token': '<bos>',
                          'eos_token': '<eos>'}

    # get the pretrained tokenizers
    tokenizer_enc = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer_dec = GPT2TokenizerFast.from_pretrained('gpt2')

    for tokenizer in [tokenizer_enc, tokenizer_dec]:
        tokenizer.add_tokens(extra_tokens, special_tokens=True)
        tokenizer.add_special_tokens(special_tokens)

    if dataset_name == 'Atomic2':
        # filename = 'atomic_preprocessed.zip'
        filename = 'atomic_simple_preprocessed.zip'
        path = os.path.join(root, filename)

        extracted_file = extract_archive(path)[0]
        with open(extracted_file) as file:
            data_dict = json.load(file)
        data_dict['valid'] = data_dict.pop('dev')

    else:
        raise ValueError(f'Data set {dataset_name} not available. Choose one of [Atomic2, Atomic2020].')

    data = dict()

    for item in data_dict.keys():
        if item not in data_select:
            continue
        data_set = defaultdict(lambda: defaultdict(dict))
        logging.info('Creating {} data'.format(item))
        _iter = iter(data_dict[item]['total'])
        id = 0
        for row in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset'):

            SOS_dec = tokenizer_dec.bos_token_id
            EOS = tokenizer_dec.eos_token_id
            PAD = tokenizer_dec.pad_token_id

            seq1 = row[0] + " " + row[1]
            seq2 = row[2]
            relation = row[1]

            tokenizer_enc_out = tokenizer_enc(seq1, seq2, return_token_type_ids=True,
                                              padding='max_length', truncation='only_second', max_length=fix_len)
            tokenizer_dec_out = tokenizer_dec(seq1 + seq2,
                                              padding='max_length', truncation=True, max_length=fix_len)

            relation = tokenizer_dec(relation)['input_ids']

            # tokens_enc = tokenizer_enc_out['input_ids'][1:]   # remove [CLS] in front
            tokens_enc = tokenizer_enc_out['input_ids']

            token_types_enc = tokenizer_enc_out['token_type_ids']
            attn_mask_enc = tokenizer_enc_out['attention_mask']

            tokens_dec = tokenizer_dec_out['input_ids']
            attn_mask_dec = tokenizer_dec_out['attention_mask']
            first_pad_idx = tokens_dec.index(PAD)

            relation_idx = tokens_dec.index(relation[0])

            if first_pad_idx + 1 < min_len:
                continue

            data_set[id]['input_enc'] = tokens_enc
            data_set[id]['input_dec'] = [SOS_dec] + tokens_dec[:-1]
            data_set[id]['target'] = tokens_dec
            data_set[id]['target'][first_pad_idx] = EOS
            data_set[id]['length'] = first_pad_idx + 1
            data_set[id]['mask_subject_relation'] = [1 if i > relation_idx else 0 for i in range(fix_len)]
            data_set[id]['token_type_ids'] = token_types_enc
            data_set[id]['attn_mask_enc'] = attn_mask_enc
            data_set[id]['attn_mask_dec'] = [1] + attn_mask_dec[:-1]
            data_set[id]['relation'] = relation

            id += 1
        data[item] = data_set
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))

    num_added_tokens = len(special_tokens) + len(extra_tokens)

    return tuple(AtomicDatasetPretrained(data[d], tokenizer_dec, num_added_tokens) for d in data_select)

def Atomic2(*args, **kwargs):
    return _setup_datasets(*(("Atomic2",) + args), **kwargs)
