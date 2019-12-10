import numpy
import torch.utils.data


class ShuffledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset
        self._index_order = numpy.random.permutation(len(self._dataset))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[self._index_order[item]]
