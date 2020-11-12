import unittest
from tyche.data.experimental.datasets import YelpReviewPolarity

from nltk.tokenize import TweetTokenizer


class DatasetsTests(unittest.TestCase):
    def test_yelp_review_polarity(self):
        tokenizer = TweetTokenizer(preserve_case=False).tokenize
        train_dataset, test_dataset, valid_dataset = YelpReviewPolarity(path_to_vectors='./data',
                                                                        emb_dim='glove.6B.100d',
                                                                        voc_size=2000, min_freq=1,
                                                                        fix_len=20, min_len=2,
                                                                        tokenizer=tokenizer,
                                                                        supervised_proportion=0.9)
        assert 'input' in train_dataset[0]
        assert 'target' in train_dataset[0]
        assert 'length' in train_dataset[0]
        assert 'label' in train_dataset[0]
        assert 'supervised' in train_dataset[0]
        #############################################
        assert 'input' in test_dataset[0]
        assert 'target' in test_dataset[0]
        assert 'length' in test_dataset[0]
        assert 'label' in test_dataset[0]
        assert 'supervised' not in test_dataset[0]


if __name__ == '__main__':
    unittest.main()
