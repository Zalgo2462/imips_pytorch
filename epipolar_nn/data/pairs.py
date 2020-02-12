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
