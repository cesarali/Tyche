import unittest

from tyche.data.loader import SimpleDataLoader


class MyTestCase(unittest.TestCase):
    def test_dataloader(self) -> None:
        dataloader = SimpleDataLoader(data_path="./resources/nba_4.npy", seq_length=10000, n_steps_ahead=5,
                                      bptt_size=100, batch_size=32)
        self.assertEqual(len(dataloader.train_data_loader.dataset), 179)
        x, y = next(iter(dataloader.train_data_loader))
        self.assertEqual(x.size(), (32, 99, 100, 4))
        self.assertEqual(y.size(), (32, 99, 100, 5, 4))


if __name__ == '__main__':
    unittest.main()
