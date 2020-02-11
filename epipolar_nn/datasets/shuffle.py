import numpy
import torch.utils.data


class ShuffledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, num_samples=None):
        self._dataset = dataset

        if num_samples is None:
            num_samples = len(self._dataset)
        self._index_order = numpy.random.permutation(num_samples)

    def __len__(self):
        return len(self._index_order)

    def __getitem__(self, item):
        return self._dataset[self._index_order[item]]
