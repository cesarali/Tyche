import numpy as np
import torch
from torchtext.data.batch import Batch
from torchtext.data.iterator import Iterator, batch, pool


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)
        self.batches = filter(lambda x: len(x) == self.batch_size, self.batches)


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)
        self.batches = filter(lambda x: len(x) == self.batch_size, self.batches)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                batch = Batch(minibatch, self.dataset, self.device)
                time, seq_len = batch.time
                text = batch.text
                num_windows = int(time.size(1) / self.bptt_len)
                time = time.view(self.batch_size, num_windows, self.bptt_len)
                text = text.view(self.batch_size, num_windows, self.bptt_len, -1)
                f_w = seq_len / self.bptt_len
                p_w = seq_len % self.bptt_len
                z_w = np.clip(num_windows - f_w - 1, 0, None)
                ###
                f_w_m = map(lambda x: torch.ones(x, dtype=torch.int64) * self.bptt_len, f_w)
                f_w_m = map(lambda x, y: torch.cat((x, y)), f_w_m, p_w.view(-1, 1))
                z_w_m = map(lambda x: torch.zeros(x, dtype=torch.int64), z_w.view(-1, 1))
                f_w_m = map(lambda x, y: torch.cat((x, y)), f_w_m, z_w_m)
                seq_len = torch.stack(list(map(lambda x: x[:num_windows], f_w_m)))
                ###
                time = time.unbind(1)
                seq_len = seq_len.unbind(1)
                text = text.unbind(1)
                yield (Batch.fromvars(
                        self.dataset, self.batch_size,
                        time=(te, l),
                        text=te) for ti, te, l in zip(time, text, seq_len))


            if not self.repeat:
                return
