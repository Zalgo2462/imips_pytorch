import os
from typing import Optional, List

import torch.utils.data

from epipolar_nn.data import pairs, klt


class KITTIMonocularStereoPairs(torch.utils.data.Dataset):

    @property
    def train_sequences(self: 'KITTIMonocularStereoPairs') -> List[str]:
        raise NotImplementedError()

    @property
    def test_sequences(self: 'KITTIMonocularStereoPairs') -> List[str]:
        raise NotImplementedError()

    @property
    def raw_folder(self: 'KITTIMonocularStereoPairs') -> str:
        raise NotImplementedError()

    @property
    def processed_folder(self: 'KITTIMonocularStereoPairs') -> str:
        raise NotImplementedError()

    def __init__(self: 'KITTIMonocularStereoPairs',
                 root: str,
                 train: Optional[bool] = True,
                 download: Optional[bool] = True,
                 left_camera_only: Optional[bool] = True,
                 minimum_KLT_overlap: Optional[float] = 0.3) -> None:
        self._root_folder = os.path.abspath(root)
        self.train = train

        self._tracker = klt.Tracker()

        if download:
            self.download()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> pairs.CorrespondencePair:
        raise NotImplementedError()

    def download(self: 'KITTIMonocularStereoPairs') -> None:
        raise NotImplementedError()

    def _check_raw_exists(self: 'KITTIMonocularStereoPairs') -> bool:
        raise NotImplementedError()

    def _check_processed_exists(self: 'KITTIMonocularStereoPairs') -> bool:
        raise NotImplementedError()


class KITTIStereoPairs(torch.utils.data.Dataset):

    @property
    def train_sequences(self: 'KITTIStereoPairs') -> List[str]:
        raise NotImplementedError()

    @property
    def test_sequences(self: 'KITTIStereoPairs') -> List[str]:
        raise NotImplementedError()

    @property
    def raw_folder(self: 'KITTIStereoPairs') -> str:
        raise NotImplementedError()

    @property
    def processed_folder(self: 'KITTIStereoPairs') -> str:
        raise NotImplementedError()

    def __init__(self: 'KITTIStereoPairs',
                 root: str,
                 train: Optional[bool] = True,
                 download: Optional[bool] = True) -> None:
        self._root_folder = os.path.abspath(root)
        self.train = train

        self._tracker = klt.Tracker()

        if download:
            self.download()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> pairs.CorrespondencePair:
        raise NotImplementedError()

    def download(self: 'KITTIStereoPairs') -> None:
        raise NotImplementedError()

    def _check_raw_exists(self: 'KITTIStereoPairs') -> bool:
        raise NotImplementedError()

    def _check_processed_exists(self: 'KITTIStereoPairs') -> bool:
        raise NotImplementedError()
