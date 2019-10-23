import numpy as np
import torch
from torchtext.data import Dataset, Field
from torchtext.data.batch import Batch
from torchtext.data.iterator import Iterator


class BPTTPointIterator(Iterator):
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
        super(BPTTPointIterator, self).__init__(dataset, batch_size, **kwargs)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = Batch(minibatch, self.dataset, self.device)
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch_size: int = len(batch)
                dataset = Dataset(examples=self.dataset.examples, fields=[
                    ('time', self.dataset.fields['time']), ('mark', self.dataset.fields['mark']),
                    ('target_time', Field(use_vocab=False)), ('target_mark', Field(use_vocab=False))])
                if self.train:
                    seq_len, time, mark = self.__series_2_bptt(batch, batch_size)
                    yield (Batch.fromvars(
                            dataset, batch_size,
                            time=(ti[:, :, :2], l),
                            mark=m[:, :, 0],
                            target_time=ti[:, :, -1],
                            target_mark=m[:, :, -1]) for ti, l, m in zip(time, seq_len, mark))
                else:
                    batch.target_time = batch.time[0][:, :, 2]
                    batch.time = (batch.time[0][:, :, :2], batch.time[1])
                    batch.target_mark = batch.mark[:, :, 1]
                    batch.mark = batch.mark[:, :, 0]
                    yield batch

            if not self.repeat:
                return

    def __series_2_bptt(self, batch: Batch, batch_size: int):
        time, seq_len = batch.time
        mark = batch.mark
        seq_len = seq_len.type(torch.int32).numpy()

        num_windows = int(time.size(1) / self.bptt_len)
        time = time.view(batch_size, num_windows, self.bptt_len, -1)
        mark = mark.view(batch_size, num_windows, self.bptt_len, -1)

        f_w = seq_len / self.bptt_len
        f_w = f_w.astype(np.int)
        p_w = seq_len % self.bptt_len
        z_w = np.clip(num_windows - f_w - 1, 0, None)

        f_w_m = map(lambda x: np.ones(int(x), dtype=np.int) * self.bptt_len, f_w)
        f_w_m = map(lambda x, y: np.concatenate((x, y)), f_w_m, p_w.reshape(-1, 1))
        z_w_m = map(lambda x: np.zeros(x, dtype=np.int), z_w.reshape(-1, 1))
        f_w_m = map(lambda x, y: np.concatenate((x, y)), f_w_m, z_w_m)
        seq_len = torch.from_numpy(np.stack(list(map(lambda x: x[:num_windows], f_w_m))))
        time = time.unbind(1)
        seq_len = seq_len.unbind(1)
        mark = mark.unbind(1)

        return seq_len, time, mark


class BPTTNestedIterator(Iterator):
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

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = Batch(minibatch, self.dataset, self.device)
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                # should we have many batches or we should have one long batch with many windows
                batch_size: int = len(batch)
                dataset = Dataset(examples=self.dataset.examples, fields=[
                    ('time', self.dataset.fields['time']), ('bow', self.dataset.fields['bow']),
                    ('target_time', Field(use_vocab=False)), ('target_text', Field(use_vocab=False))])
                if self.train:
                    seq_len, text, time, bow = self.__series_2_bptt(batch, batch_size)
                    yield (Batch.fromvars(
                            dataset, batch_size,
                            time=(t[:, :, :2], l),
                            bow=b,
                            target_time=t[:, :, -1],
                            target_text=(te, sl)) for t, b, te, sl, l in zip(time, bow, text[0], text[1], seq_len))
                else:

                    batch.target_text = batch.text
                    batch.time = (batch.time[0][:, :, :2], batch.time[1])
                    batch.target_time = batch.time[0][:, :, -1]
                    yield batch

            if not self.repeat:
                return

    def __series_2_bptt(self, batch: Batch, batch_size: int):
        time, seq_len = batch.time
        text, seq_len, sentence_len = batch.text
        bow = batch.bow
        num_windows = int(time.size(1) / self.bptt_len)
        time = time.view(batch_size, num_windows, self.bptt_len, -1)
        bow = bow.view(batch_size, num_windows, self.bptt_len, -1)
        text = text.view(batch_size, num_windows, self.bptt_len, -1)
        sentence_len = sentence_len.view(batch_size, num_windows, self.bptt_len)
        f_w = seq_len / self.bptt_len
        p_w = seq_len % self.bptt_len
        z_w = np.clip(num_windows - f_w - 1, 0, None)

        f_w_m = map(lambda x: torch.ones(x, dtype=torch.int64) * self.bptt_len, f_w)
        f_w_m = map(lambda x, y: torch.cat((x, y)), f_w_m, p_w.view(-1, 1))
        z_w_m = map(lambda x: torch.zeros(x, dtype=torch.int64), z_w.view(-1, 1))
        f_w_m = map(lambda x, y: torch.cat((x, y)), f_w_m, z_w_m)
        seq_len = torch.stack(list(map(lambda x: x[:num_windows], f_w_m)))
        time = time.unbind(1)
        seq_len = seq_len.unbind(1)
        bow = bow.unbind(1)
        text = (text.unbind(1), sentence_len.unbind(1))
        return seq_len, text, time, bow


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

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = Batch(minibatch, self.dataset, self.device)
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                # should we have many batches or we should have one long batch with many windows
                batch_size: int = len(batch)
                dataset = Dataset(examples=self.dataset.examples, fields=[
                    ('time', self.dataset.fields['time']), ('bow', self.dataset.fields['bow']),
                    ('target_time', Field(use_vocab=False)), ('target_bow', Field(use_vocab=False))])
                if self.train:
                    seq_len, time, bow = self.__series_2_bptt(batch, batch_size)
                    yield (Batch.fromvars(
                            dataset, batch_size,
                            time=(t[:, :, :2], l),
                            bow=b[:, :, 0],
                            target_time=t[:, :, -1],
                            target_bow=b[:, :, 0]) for t, b, l in zip(time, bow, seq_len))
                else:
                    batch.target_bow = batch.bow[:, :, 1]
                    batch.bow = batch.bow[:, :, 0]
                    batch.target_time = batch.time[0][:, :, -1]
                    batch.time = (batch.time[0][:, :, :2], batch.time[1])
                    yield batch

            if not self.repeat:
                return

    def __series_2_bptt(self, batch: Batch, batch_size: int):
        time, seq_len = batch.time
        bow = batch.bow
        num_windows = int(time.size(1) / self.bptt_len)
        time = time.view(batch_size, num_windows, self.bptt_len, -1)
        bow = bow.view(batch_size, num_windows, self.bptt_len, 2, -1)
        f_w = seq_len / self.bptt_len
        p_w = seq_len % self.bptt_len
        z_w = np.clip(num_windows - f_w - 1, 0, None)

        f_w_m = map(lambda x: torch.ones(x, dtype=torch.int64) * self.bptt_len, f_w)
        f_w_m = map(lambda x, y: torch.cat((x, y)), f_w_m, p_w.view(-1, 1))
        z_w_m = map(lambda x: torch.zeros(x, dtype=torch.int64), z_w.view(-1, 1))
        f_w_m = map(lambda x, y: torch.cat((x, y)), f_w_m, z_w_m)
        seq_len = torch.stack(list(map(lambda x: x[:num_windows], f_w_m)))
        time = time.unbind(1)
        seq_len = seq_len.unbind(1)
        bow = bow.unbind(1)

        return seq_len, time, bow
