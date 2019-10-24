from torchtext import data

make_example = data.Example.fromdict


def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')


class PennTreebank(data.Dataset):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.
    References
    ----------
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    name = 'penn-treebank'
    dirname = ''

    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        pos = data.TabularDataset(path, format='csv', fields=fields)
        super(PennTreebank, self).__init__(
                pos.examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, root='.data', train='ptb.train.txt',
               validation='ptb.valid.txt', test='ptb.test.txt',
               **kwargs):
        """Create dataset objects for splits of the Penn Treebank dataset.
        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """
        return super(PennTreebank, cls).splits(
                root=root, train=train, validation=validation, test=test,
                text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, text_field, batch_size=32, device='cpu', root='.data',
              vectors=None, vectors_cache=None, max_size=None, min_freq=1, **kwargs):
        """Create iterator objects for splits of the Penn Treebank dataset.
        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.
        Arguments:
            text_field: The field that will be used for text data.
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory where the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        train, val, test = cls.splits(text_field, root=root, **kwargs)

        text_field.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache, max_size=max_size,
                               min_freq=min_freq)

        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)

    @property
    def max_len(self):
        return max([len(f.text) for f in self.examples])


class WikiText2(data.Dataset):
    urls = ['https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip']
    name = 'wikitext-2'
    dirname = 'wikitext-2'

    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        pos = data.TabularDataset(path, format='csv', fields=fields)

        super(WikiText2, self).__init__(
                pos.examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, root='.data', train='wiki.train.tokens',
               validation='wiki.valid.tokens', test='wiki.test.tokens',
               **kwargs):
        """Create dataset objects for splits of the WikiText-2 dataset.

              This is the most flexible way to use the dataset.

              Arguments:
                  text_field: The field that will be used for text data.
                  root: The root directory that the dataset's zip archive will be
                      expanded into; therefore the directory in whose wikitext-2
                      subdirectory the data files will be stored.
                  train: The filename of the train data. Default: 'wiki.train.tokens'.
                  validation: The filename of the validation data, or None to not
                      load the validation set. Default: 'wiki.valid.tokens'.
                  test: The filename of the test data, or None to not load the test
                      set. Default: 'wiki.test.tokens'.
              """
        return super(WikiText2, cls).splits(
                root=root, train=train, validation=validation, test=test,
                text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, text_field, batch_size=32, device='cpu', root='.data',
              vectors=None, vectors_cache=None, max_size=None, min_freq=1, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.

              This is the simplest way to use the dataset, and assumes common
              defaults for field, vocabulary, and iterator parameters.

              Arguments:
                  batch_size: Batch size.
                  bptt_len: Length of sequences for backpropagation through time.
                  device: Device to create batches on. Use -1 for CPU and None for
                      the currently active GPU device.
                  root: The root directory that the dataset's zip archive will be
                      expanded into; therefore the directory in whose wikitext-2
                      subdirectory the data files will be stored.
                  wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                      text field. The word vectors are accessible as
                      train.dataset.fields['text'].vocab.vectors.
                  Remaining keyword arguments: Passed to the splits method.
              """
        train, val, test = cls.splits(text_field, root=root, **kwargs)

        text_field.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache, max_size=max_size,
                               min_freq=min_freq)

        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)

    @property
    def max_len(self):
        return max([len(f.text) for f in self.examples])


class WikiText103(data.Dataset):
    urls = ['https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip']
    name = 'wikitext-103'
    dirname = 'wikitext-103'

    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        pos = data.TabularDataset(path, format='csv', fields=fields)

        super(WikiText103, self).__init__(
                pos.examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, root='.data', train='wiki.train.tokens',
               validation='wiki.valid.tokens', test='wiki.test.tokens',
               **kwargs):
        """Create dataset objects for splits of the WikiText-103 dataset.

              This is the most flexible way to use the dataset.

              Arguments:
                  text_field: The field that will be used for text data.
                  root: The root directory that the dataset's zip archive will be
                      expanded into; therefore the directory in whose wikitext-2
                      subdirectory the data files will be stored.
                  train: The filename of the train data. Default: 'wiki.train.tokens'.
                  validation: The filename of the validation data, or None to not
                      load the validation set. Default: 'wiki.valid.tokens'.
                  test: The filename of the test data, or None to not load the test
                      set. Default: 'wiki.test.tokens'.
              """
        return super(WikiText103, cls).splits(
                root=root, train=train, validation=validation, test=test,
                text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, text_field, batch_size=32, device='cpu', root='.data',
              vectors=None, vectors_cache=None, max_size=None, min_freq=1, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.

              This is the simplest way to use the dataset, and assumes common
              defaults for field, vocabulary, and iterator parameters.

              Arguments:
                  batch_size: Batch size.
                  bptt_len: Length of sequences for backpropagation through time.
                  device: Device to create batches on. Use -1 for CPU and None for
                      the currently active GPU device.
                  root: The root directory that the dataset's zip archive will be
                      expanded into; therefore the directory in whose wikitext-2
                      subdirectory the data files will be stored.
                  wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                      text field. The word vectors are accessible as
                      train.dataset.fields['text'].vocab.vectors.
                  Remaining keyword arguments: Passed to the splits method.
              """
        train, val, test = cls.splits(text_field, root=root, **kwargs)

        text_field.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache, max_size=max_size,
                               min_freq=min_freq)

        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)

    @property
    def max_len(self):
        return max([len(f.text) for f in self.examples])
