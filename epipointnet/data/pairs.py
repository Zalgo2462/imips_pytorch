from abc import ABC
from typing import Tuple

import numpy as np


class ImagePair(ABC):
    @property
    def image_1(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def image_2(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()


class CorrespondencePair(ImagePair, ABC):

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class FundamentalMatrixPair(ImagePair, ABC):

    @property
    def f_matrix_forward(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def f_matrix_backward(self) -> np.ndarray:
        raise NotImplementedError()


class CorrespondenceFundamentalMatrixPair(CorrespondencePair, FundamentalMatrixPair):

    def __init__(self, correspondence_pair: CorrespondencePair, fundamental_matrix_pair: FundamentalMatrixPair):
        self._corr_pair = correspondence_pair
        self._f_pair = fundamental_matrix_pair
        assert self._corr_pair.name is self._f_pair.name
        assert self._corr_pair.image_1 is self._f_pair.image_1
        assert self._corr_pair.image_2 is self._f_pair.image_2

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return self._corr_pair.correspondences(pixels_xy, inverse)

    @property
    def f_matrix_forward(self) -> np.ndarray:
        return self._f_pair.f_matrix_forward

    @property
    def f_matrix_backward(self) -> np.ndarray:
        return self._f_pair.f_matrix_backward

    @property
    def image_1(self) -> np.ndarray:
        return self._f_pair.image_1

    @property
    def image_2(self) -> np.ndarray:
        return self._f_pair.image_2

    @property
    def name(self) -> str:
        return self._f_pair.name
