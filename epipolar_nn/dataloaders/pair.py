import numpy as np


class ImagePair:

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> np.ndarray:
        pass

    @property
    def image_1(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def image_2(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()
