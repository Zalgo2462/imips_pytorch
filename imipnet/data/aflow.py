from typing import Tuple

import numpy as np

from imipnet.data.pairs import CorrespondencePair


class AbsoluteFlowPair(CorrespondencePair):
    def __init__(self, image_1: np.ndarray, image_2: np.ndarray, name: str,
                 absolute_forward_flow: np.ndarray, absolute_backward_flow: np.ndarray):
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__name = name
        self.__forward_flow = absolute_forward_flow
        self.__backward_flow = absolute_backward_flow

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        flow = self.__forward_flow if not inverse else self.__backward_flow

        pixels_xy = np.round(pixels_xy).astype(np.int)
        pixels_xy[0, pixels_xy[0] == flow.shape[2]] -= 1
        pixels_xy[1, pixels_xy[1] == flow.shape[1]] -= 1

        results = flow[:, pixels_xy[1], pixels_xy[0]]
        indices = np.arange(pixels_xy.shape[1])
        mask = ~np.isnan(results[0])

        corr_pixels_xy = results[:, mask]
        tracked_indices = indices[mask]

        return corr_pixels_xy, tracked_indices

    @property
    def image_1(self) -> np.ndarray:
        return self.__image_1

    @property
    def image_2(self) -> np.ndarray:
        return self.__image_2

    @property
    def name(self) -> str:
        return self.__name
