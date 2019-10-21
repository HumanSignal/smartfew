import io

import numpy as np


class UnseenStringSamplingProvider(object):

    def __init__(self, items, batch_size, random_seed=42):
        self.items = np.array(items, dtype=np.object)
        self.batch_size = batch_size
        self.random_seed = random_seed

        self._seen = np.zeros(len(items), dtype=np.bool)
        self._rng = np.random.RandomState(self.random_seed)

    def __iter__(self):
        self._seen[:] = False
        return self

    def __next__(self):
        unseen_idx = np.where(np.logical_not(self._seen))[0]
        if len(unseen_idx) > 0:
            self._rng.shuffle(unseen_idx)
            chosen_idx = unseen_idx[:self.batch_size]
            self._seen[chosen_idx] = True
            return self.items[chosen_idx].tolist()
        else:
            raise StopIteration


class ImageURLsFromFileProvider(UnseenStringSamplingProvider):

    def __init__(self, urllist_file, batch_size=1000):
        self.urllist_file = urllist_file
        with io.open(self.urllist_file) as f:
            items = np.array(f.read().splitlines(), dtype=np.object)
        super(ImageURLsFromFileProvider, self).__init__(items, batch_size)

