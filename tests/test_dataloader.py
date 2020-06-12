import unittest
from itertools import islice
from tyche.data.experimental.loader import DataLoaderWiki2, DataLoaderPTB


class MyTestCase(unittest.TestCase):
    # def test_dataloader(self) -> None:
    #     dataloader = SimpleDataLoader(data_path="./resources/nba_4.npy", seq_length=10000, n_steps_ahead=5,
    #                                   bptt_size=100, batch_size=32)
    #     self.assertEqual(len(dataloader.train_data_loader.dataset), 179)
    #     x, y = next(iter(dataloader.train_data_loader))
    #     self.assertEqual(x.size(), (32, 99, 100, 4))
    #     self.assertEqual(y.size(), (32, 99, 100, 5, 4))

    def test_wiki2_dataloader(self):
        dl = DataLoaderWiki2('cpu', path_to_data='./data/', batch_size=32, path_to_vectors='./data', emb_dim='glove.6B.100d', voc_size=2000, min_freq=1,
                             fix_len=20)
        for batch in islice(dl.train, 3):
            self.assertEqual(batch[0].size(), (32, 22))
            self.assertEqual(batch[1].size(), (32,))

    def test_ptb_dataloader(self):
        dl = DataLoaderPTB('cpu', path_to_data='./data/', batch_size=32, path_to_vectors='./data',
                           emb_dim='glove.6B.100d', voc_size=2000, min_freq=1,
                           fix_len=20)
        for batch in islice(dl.train, 3):
            self.assertEqual(batch[0].size(), (32, 22))
            self.assertEqual(batch[1].size(), (32,))


if __name__ == '__main__':
    unittest.main()
