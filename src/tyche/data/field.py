import torch
from torchtext.data import Field


class BPTTField(Field):
    def __init__(self, bptt_len=20, **kwargs):
        super(BPTTField, self).__init__(sequential=True, batch_first=True, **kwargs)
        self.bptt_len = bptt_len
        
    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch

        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2

        reminder = max_len % self.bptt_len
        if reminder != 0:
            max_len += (self.bptt_len - reminder)
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                        [self.pad_token] * max(0, max_len - len(x)) +
                        ([] if self.init_token is None else [self.init_token]) +
                        list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                        ([] if self.init_token is None else [self.init_token]) +
                        list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]) +
                        [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded


class ReversibleField(Field):
    def __init__(self, **kwargs):
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch):

        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]

        return [' '.join(ex) for ex in batch]
