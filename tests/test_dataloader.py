import unittest
from itertools import islice
from tyche.data.experimental.loader import (
    DataLoaderAtomic2FinetunedKL,
    DataLoaderWiki2,
    DataLoaderPTB,
    DataLoaderWiki103,
    DataLoaderYelpReviewPolarity,
)
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_wiki2_dataloader(self):
        dl = DataLoaderWiki2(
            "cpu",
            path_to_data="./data/",
            batch_size=32,
            path_to_vectors="./data",
            emb_dim="glove.6B.100d",
            voc_size=2000,
            min_freq=1,
            fix_len=20,
            min_len=2,
        )
        for batch in islice(dl.train, 3):
            self.assertEqual(batch["input"].size(), (32, 20))
            self.assertEqual(batch["length"].size(), (32,))

    def test_wiki103_dataloader(self):
        dl = DataLoaderWiki103(
            "cpu",
            path_to_data="./data/",
            batch_size=32,
            path_to_vectors="./data",
            emb_dim="glove.6B.100d",
            voc_size=2000,
            min_freq=1,
            fix_len=20,
            min_len=2,
        )
        for batch in islice(dl.train, 3):
            self.assertEqual(batch["input"].size(), (32, 20))
            self.assertEqual(batch["length"].size(), (32,))

    def test_ptb_dataloader(self):
        dl = DataLoaderPTB(
            "cpu",
            path_to_data="./data/",
            batch_size=32,
            path_to_vectors="./data",
            emb_dim="glove.6B.100d",
            voc_size=2000,
            min_freq=1,
            fix_len=20,
            min_len=2,
        )
        for batch in islice(dl.train, 3):
            self.assertEqual(batch["input"].size(), (32, 20))
            self.assertEqual(batch["length"].size(), (32,))

    def test_yelp_polarity_dataloader(self):
        dl = DataLoaderYelpReviewPolarity(
            "cpu",
            path_to_data="./data/",
            batch_size=32,
            path_to_vectors="./data",
            emb_dim="glove.6B.100d",
            voc_size=2000,
            min_freq=1,
            fix_len=20,
            min_len=2,
            supervised_proportion=0.9,
        )
        s = []
        for batch in islice(dl.train, 10000):
            s.append(batch["supervised"].sum())
            self.assertEqual(batch["input"].size(), (32, 20))
            self.assertEqual(batch["length"].size(), (32,))
        print(np.mean(s))

    def test_Atomic2FinetundeKL_dataloader(self):
 
        args = {
            "add_gen_token": False,
            "batch_size": 32,
            "fix_len": 48,
            "min_len": 1,
            "path_to_data": "/raid/data/gentext/atomic2",
            "path_to_pretrained_models": None,
            "get_test_unshuffled": True,
        }

        dl = DataLoaderAtomic2FinetunedKL("cuda:0", **args)
        for batch in islice(dl.train, 3):
            self.assertEqual(batch["input"].size(), (32, 20))
            self.assertEqual(batch["length"].size(), (32,))


if __name__ == "__main__":
    unittest.main()
