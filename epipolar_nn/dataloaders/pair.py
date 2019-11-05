import numpy as np


class StereoPair:

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> np.ndarray:
        raise NotImplementedError()

    @property
    def image_1(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def image_2(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()


class HomographyPair(StereoPair):
    def __init__(self, image_1: np.ndarray, image_2: np.ndarray,
                 homography: np.ndarray, name: str):
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__name = name
        self._H = homography
        self._inv_H = np.linalg.inv(homography)

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> np.ndarray:
        # pixels_xy are a 2d column major array
        tx_h = self._H
        if inverse:
            tx_h = self._inv_H

        homogeneous_tx_points = np.dot(
            tx_h,
            np.vstack((
                pixels_xy,
                np.ones((1, pixels_xy.shape[1]))
            ))
        )
        return homogeneous_tx_points[0:2, :] / homogeneous_tx_points[2, :]

    @property
    def image_1(self) -> np.ndarray:
        return self.__image_1

    @property
    def image_2(self) -> np.ndarray:
        return self.__image_2

    @property
    def name(self) -> str:
        return self.__name
