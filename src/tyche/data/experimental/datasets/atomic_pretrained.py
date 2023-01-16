import csv
import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from torchtext.utils import download_from_url, extract_archive
import torch
import gzip
from tqdm import tqdm
from transformers import BertTokenizer, GPT2Tokenizer
from tyche.data.experimental.datasets.language_modeling import LanguageModelingDataset

URLS = {"Atomic2020": "https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip",
        'ConceptNet': ['https://ttic.uchicago.edu/~kgimpel/comsense_resources/train300k.txt.gz',
                       'https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz',
                       'https://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz'],
        'ConceptNet5': 'https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz'}


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

    def __init__(self, data, tokenizer_list, num_added_tokens, num_relations):
        """Initiate language modeling dataset using pretrained tokenizers from huggingface.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            tokenizer_list: list of huggingface tokenizers
            num_added_tokens: number of tokens that were newly added to the vocabulary of each tokenizer
        """

        super(AtomicDatasetPretrained, self).__init__(data, tokenizer_list)
        self.data = data
        self.tokenizer_list = tokenizer_list
        self.tokenizer = tokenizer_list[-1]

        self.num_added_tokens = num_added_tokens
        self.num_relations = num_relations

    def __getitem__(self, i):
        return {
            "id": i,
            "input_enc": np.asarray(self.data[i]["input_enc"], dtype=np.int64),
            "input_dec": np.asarray(self.data[i]["input_dec"], dtype=np.int64),
            "target": np.asarray(self.data[i]["target"], dtype=np.int64),
            "length": np.asarray(self.data[i]["length"]),
            "token_type_ids": np.asarray(self.data[i]["token_type_ids"]),
            "attn_mask_enc": np.asarray(self.data[i]["attn_mask_enc"]),
            "attn_mask_dec": np.asarray(self.data[i]["attn_mask_dec"]),
            "mask_subject_relation": np.asarray(self.data[i]["mask_subject_relation"]),
            "relation": np.asarray(self.data[i]["relation"]),
        }

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def get_pad_token_id(self):
        return -100

    def get_unk_token_id(self):
        return self.tokenizer.unk_token_id

    def get_gen_token_id(self):
        return self.tokenizer.convert_tokens_to_ids("[GEN]")

    def get_num_added_tokens(self):
        return self.num_added_tokens

    def reverse(self, batch):
        batch[batch == -100] = self.tokenizer.eos_token_id
        sentences = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
        return sentences


class AtomicDatasetFinetunedKL(AtomicDatasetPretrained):
    def __init__(self, data, tokenizer_list, num_added_tokens, path_to_posterior_samples: str):
        super().__init__(data, tokenizer_list, num_added_tokens)
        self.path_to_posterior_samples = Path(path_to_posterior_samples)
        self.posterior_samples = h5py.File(self.path_to_posterior_samples, "r")

    def __getitem__(self, i):

        return {
            "input_enc": np.asarray(self.data[i]["input_enc"], dtype=np.int64),
            "token_type_ids": np.asarray(self.data[i]["token_type_ids"]),
            "attn_mask_enc": np.asarray(self.data[i]["attn_mask_enc"]),
            "walks": self.posterior_samples[f"{i}/walks"][:],
            "walks_prob": self.posterior_samples[f"{i}/walks_prob"][:],
            "q_matrix": self.posterior_samples[f"{i}/q_matrix"][:],
        }


def _setup_datasets(
    dataset_name,
    fix_len,
    min_len=0,
    min_freq=1,
    root="./data",
    data_select=("train", "test", "valid", 'test_unshuffled_unique'),
    add_gen_token=False,
    path_to_posterior_samples=None,
):
    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset({"train", "test", "valid", 'test_unshuffled_unique'}):
        raise TypeError("data_select is not supported!")

    Path(root).mkdir(parents=True, exist_ok=True)
    if dataset_name == "Atomic2":
        # filename = 'atomic_preprocessed.zip'
        filename = "atomic_simple_preprocessed.zip"
        path = os.path.join(root, filename)

        extracted_file = extract_archive(path)[0]
        with open(extracted_file) as file:
            data_dict = json.load(file)
        data_dict["valid"] = data_dict.pop("dev")

    elif dataset_name == "Atomic2020":
        filename = "atomic2020_extracted"
        path = os.path.join(root, filename)

        # download and extract if not present
        if not os.path.exists(path):
            url = URLS[dataset_name]
            zip_path = os.path.join(root, "atomic2020.zip")
            download_from_url(url, path=zip_path, overwrite=True)
            extract_archive(zip_path, to_path=path)
        path = os.path.join(path, "atomic2020_data-feb2021")
        data_dict = dict()
        data_dict["train"] = csv.reader(open(os.path.join(path, "train.tsv")), delimiter="\t")
        data_dict["test"] = csv.reader(open(os.path.join(path, "test.tsv")), delimiter="\t")
        data_dict["valid"] = csv.reader(open(os.path.join(path, "dev.tsv")), delimiter="\t")

    elif dataset_name == 'ConceptNet':
        filenames = ['train300k.txt.gz', 'dev1.txt.gz', 'test.txt.gz']
        setnames = ['train', 'valid', 'test']
        data_dict = dict()
        for i, (filename, setname) in enumerate(zip(filenames, setnames)):
            path = os.path.join(root, filename)

            # download and extract if not present
            if not os.path.exists(path):
                url = URLS[dataset_name][i]
                download_from_url(url, path=path, overwrite=True)
            file = gzip.open(path, 'rt')
            data_dict[setname] = csv.reader(file, delimiter="\t")

    elif dataset_name == 'ConceptNet5':
        filename = 'conceptnet-assertions-5.7.0-english.csv'

        # download and extract if not present
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            zip_path = os.path.join(root, 'conceptnet-assertions-5.7.0.csv.gz')
            url = URLS[dataset_name]
            download_from_url(url, path=zip_path, overwrite=True)
            with gzip.open(zip_path, 'rt') as in_file:
                data_file = csv.reader(in_file, delimiter='\t')
                with open(path, 'w') as out_file:
                    writer = csv.writer(out_file)
                    print('filtering out english examples')
                    for line in tqdm(data_file):
                        # only english and no dbpedia
                        if line[2][3:5] == line[3][3:5] == 'en' and 'dbpedia' not in line[1]:
                            writer.writerow(['<' + line[1][3:] + '>', line[2][6:-2], line[3][6:]])

        file = open(path, 'rt')
        data_file = np.array(list(csv.reader(file)))
        data_size = len(data_file)
        print(data_file[:10])

        # randomly permute the data
        rng = np.random.default_rng(seed=0)
        data_file = data_file[rng.permutation(data_size)]

        # determine train (90%) and valid/test (5%/5%) sizes
        train_size = int(data_size * .9)
        test_size = (data_size - train_size) // 2

        # create 3 splits from the data
        data_dict = dict()
        data_dict["train"] = data_file[:train_size]
        data_dict["test"] = data_file[train_size:train_size+test_size]
        data_dict["valid"] = data_file[train_size+test_size:]

        # determine extra tokens
        relations = list(map(list, zip(*data_file)))[0]
        extra_tokens = list(set(relations))

    else:
        raise ValueError(f"Data set {dataset_name} not available. Choose one of [Atomic2, Atomic2020, ConceptNet, ConceptNet5].")

    if dataset_name == "Atomic2":
        extra_tokens = [
            "<oEffect>",
            "<oReact>",
            "<oWant>",
            "<xAttr>",
            "<xEffect>",
            "<xIntent>",
            "<xNeed>",
            "<xReact>",
            "<xWant>",
            "none",
            "___",
            "PersonX",
            "PersonY",
        ]
    elif dataset_name == "Atomic2020":
        extra_tokens = [
            "<AtLocation>",
            "<CapableOf>",
            "<Causes>",
            "<CausesDesire>",
            "<CreatedBy>",
            "<Desires>",
            "<HasA>",
            "<HasFirstSubevent>",
            "<HasLastSubevent>",
            "<HasPrerequisite>",
            "<HasProperty>",
            "<HasSubEvent>",
            "<HinderedBy>",
            "<InstanceOf>",
            "<isAfter>",
            "<isBefore>",
            "<isFilledBy>",
            "<MadeOf>",
            "<MadeUpOf>",
            "<MotivatedByGoal>",
            "<NotDesires>",
            "<ObjectUse>",
            "<oEffect>",
            "<oReact>",
            "<oWant>",
            "<PartOf>",
            "<ReceivesAction>",
            "<xAttr>",
            "<xEffect>",
            "<xIntent>",
            "<xNeed>",
            "<xReact>",
            "<xReason>",
            "<xWant>",
            "none",
            "___",
            "PersonX",
            "PersonY",
        ]
    elif dataset_name == 'ConceptNet':
        extra_tokens = ['<AtLocation>',
                        '<CapableOf>',
                        '<Causes>',
                        '<CausesDesire>',
                        '<CreatedBy>',
                        '<DefinedAs>',
                        '<DesireOf>',
                        '<Desires>',
                        '<HasA>',
                        '<HasFirstSubevent>',
                        '<HasLastSubevent>',
                        '<HasPainCharacter>',
                        '<HasPainIntensity>',
                        '<HasPrerequisite>',
                        '<HasProperty>',
                        '<HasSubevent>',
                        '<InheritsFrom>',
                        '<InstanceOf>',
                        '<IsA>',
                        '<LocatedNear>',
                        '<LocationOfAction>',
                        '<MadeOf>',
                        '<MotivatedByGoal>',
                        '<NotCapableOf>',
                        '<NotDesires>',
                        '<NotHasA>',
                        '<NotHasProperty>',
                        '<NotIsA>',
                        '<NotMadeOf>',
                        '<PartOf>',
                        '<ReceivesAction>',
                        '<RelatedTo>',
                        '<SymbolOf>',
                        '<UsedFor>']

    relation_eye = torch.eye(len(extra_tokens))

    # get the pretrained tokenizers
    tokenizer_enc = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_dec = GPT2Tokenizer.from_pretrained("gpt2")

    num_added_t_enc = tokenizer_enc.add_tokens(extra_tokens)
    if add_gen_token:
        num_added_t_dec = tokenizer_dec.add_tokens(extra_tokens + ["[GEN]"])
    else:
        num_added_t_dec = tokenizer_dec.add_tokens(extra_tokens)

    data = dict()
    data_dict['test_unshuffled_unique'] = data_dict['test']
    for item in data_dict.keys():
        if item not in data_select:
            continue
        preprocessed_path = os.path.join(root, f"preprocessed_len{fix_len}_{item}_gentok-{add_gen_token}.pkl")
        if os.path.exists(preprocessed_path):
            with open(preprocessed_path, "rb") as file:
                data[item] = pickle.load(file)
            print(f"loading {item} data from {preprocessed_path}")
            continue
        data_set = defaultdict(lambda: defaultdict(dict))
        logging.info("Creating {} data".format(item))
        if dataset_name == "Atomic2":
            _iter = iter(data_dict[item]["total"])
        elif dataset_name in ['Atomic2020', 'ConceptNet', 'ConceptNet5']:
            _iter = data_dict[item]
        id = 0
        unique_sub_rel = []
        for row in tqdm(_iter, unit="data point", desc=f"Preparing {item} dataset"):

            SOS = tokenizer_dec.bos_token_id
            EOS = tokenizer_dec.eos_token_id
            PAD = -100
            GEN = tokenizer_dec.convert_tokens_to_ids("[GEN]")

            if dataset_name == "Atomic2":
                relation = " " + row[1]
                subject = row[0]
                object = row[2]
            elif dataset_name == "Atomic2020":
                relation = " <" + row[1] + ">"
                subject = row[0]
                object = row[2]
            elif dataset_name == 'ConceptNet':
                relation = ' <' + row[0] + '>'
                subject = row[1]
                object = row[2]
                # skip negative examples
                if item != 'train' and row[3] == '0':
                    continue
            elif dataset_name == 'ConceptNet5':
                relation = ' ' + row[0]
                subject = row[1]
                object = row[2]
            seq1 = subject + relation
            seq2 = " " + object

            if item == 'test_unshuffled_unique':
                if subject + relation in unique_sub_rel:
                    continue
                else:
                    unique_sub_rel.append(subject + relation)
            ### Encoder ###
            tokenizer_enc_out = tokenizer_enc(
                seq1, seq2, return_token_type_ids=True, padding="max_length", truncation="only_second", max_length=fix_len
            )

            # tokens_enc = tokenizer_enc_out['input_ids'][1:]   # remove [CLS] in front
            tokens_enc = tokenizer_enc_out["input_ids"]

            token_types_enc = tokenizer_enc_out["token_type_ids"]
            attn_mask_enc = tokenizer_enc_out["attention_mask"]

            ### Decoder ###
            tokenizer_dec_out = tokenizer_dec(seq1 + seq2, truncation=True, max_length=fix_len - 2, return_length=True)
            relation_token = tokenizer_dec(relation)["input_ids"]

            length = tokenizer_dec_out["length"]
            tokens_dec = tokenizer_dec_out["input_ids"]
            attn_mask_dec = tokenizer_dec_out["attention_mask"]
            relation_idx = tokens_dec.index(relation_token[0])

            relation_one_hot = relation_eye[extra_tokens.index(relation[1:])]

            if add_gen_token:
                mask_sr_len = relation_idx + 1
                tokens_dec = tokens_dec[:mask_sr_len] + [GEN] + tokens_dec[mask_sr_len:]
                pad_length = fix_len - length - 2
                front_pad = 2
            else:
                pad_length = fix_len - length - 1
                front_pad = 1
                mask_sr_len = relation_idx

            if length < min_len:
                continue

            data_set[id]["input_enc"] = tokens_enc
            data_set[id]["input_dec"] = [SOS] + tokens_dec + pad_length * [EOS]
            data_set[id]["target"] = tokens_dec + [EOS] + pad_length * [PAD]
            data_set[id]["length"] = length + 2 if add_gen_token else length + 1
            data_set[id]["mask_subject_relation"] = [1 if i > mask_sr_len else 0 for i in range(fix_len)]
            data_set[id]["token_type_ids"] = token_types_enc
            data_set[id]["attn_mask_enc"] = attn_mask_enc
            data_set[id]["attn_mask_dec"] = [1] * front_pad + attn_mask_dec + pad_length * [0]
            data_set[id]["relation"] = relation_one_hot

            id += 1
        data[item] = data_set
        # save data dict to disk
        with open(preprocessed_path, "wb+") as file:
            pickle.dump(dict(data_set), file)
    for key in data_select:
        if not data[key]:
            raise TypeError("Dataset {} is empty!".format(key))
    if path_to_posterior_samples is None:
        return tuple(
            AtomicDatasetPretrained(data[d], [tokenizer_enc, tokenizer_dec], (num_added_t_enc, num_added_t_dec), len(extra_tokens))
            for d in data_select
        )
    else:
        return tuple(
            AtomicDatasetFinetunedKL(
                data[d],
                [tokenizer_enc, tokenizer_dec],
                (num_added_t_enc, num_added_t_dec),
                path_to_posterior_samples + "/posterior_outputs_" + d + ".hdf5",
            )
            for d in data_select
        )


def Atomic2(*args, **kwargs):
    return _setup_datasets(*(("Atomic2",) + args), **kwargs)


def Atomic2020(*args, **kwargs):
    return _setup_datasets(*(("Atomic2020",) + args), **kwargs)

def ConceptNet(*args, **kwargs):
    return _setup_datasets(*(("ConceptNet",) + args), **kwargs)

def ConceptNet5(*args, **kwargs):
    return _setup_datasets(*(("ConceptNet5",) + args), **kwargs)
