import bisect
from abc import ABC
from typing import Sequence, Tuple

import numpy as np

from epipolar_nn.data import klt


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


class HomographyPair(CorrespondencePair):
    def __init__(self, image_1: np.ndarray, image_2: np.ndarray,
                 homography: np.ndarray, name: str):
        self.__image_1 = image_1
        self.__image_2 = image_2
        self.__name = name
        self._H = homography
        self._inv_H = np.linalg.inv(homography)

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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

        corr_pixels_xy = homogeneous_tx_points[0:2, :] / homogeneous_tx_points[2, :]
        tracked_indices = np.arange(pixels_xy.shape[1])
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


class KLTPair(CorrespondencePair):

    def __init__(self, images: Sequence[np.ndarray], tracker: klt.Tracker, name: str):
        self.__sequence = list(images)  # Ensure the sequence is loaded into memory
        self.__name = name
        self._tracker = tracker

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = len(self.__sequence)
        iter_order = range(0, seq_length)
        if inverse:
            iter_order = iter_order[::-1]

        prev_img = self.__sequence[iter_order[0]]

        keypoints_rc = np.flipud(pixels_xy)
        keypoint_indices = np.arange(keypoints_rc.shape[1], dtype=int)
        for i_iter in range(1, seq_length):
            i = iter_order[i_iter]
            curr_img = self.__sequence[i]
            tracked_status, keypoints_rc = self._tracker.track(
                prev_img, curr_img, keypoints_rc
            )
            keypoint_indices = keypoint_indices[tracked_status == klt.Tracker.TRACK_SUCCESS]
            prev_img = curr_img

        keypoints_xy = np.flipud(keypoints_rc)
        return keypoints_xy, keypoint_indices

    @property
    def image_1(self) -> np.ndarray:
        return self.__sequence[0]

    @property
    def image_2(self) -> np.ndarray:
        return self.__sequence[-1]

    @property
    def name(self) -> str:
        return self.__name


class KLTPairGenerator:

    def __init__(self: 'KLTPairGenerator', name: str, images: Sequence[np.ndarray],
                 tracker: klt.Tracker, sequence_overlap: klt.SequenceOverlap, minimum_overlap: float):
        self.name = name
        self.img_sequence = images
        self._tracker = tracker
        self._min_overlap = minimum_overlap
        self._overlapped_frames_cum_sum = []

        overlapped_frames = sequence_overlap.find_frames_with_overlap(0, self._min_overlap).size
        self._overlapped_frames_cum_sum.append(overlapped_frames)
        for i in range(1, len(self.img_sequence) - 1):
            overlapped_frames = sequence_overlap.find_frames_with_overlap(i, self._min_overlap).size
            self._overlapped_frames_cum_sum.append(overlapped_frames + self._overlapped_frames_cum_sum[-1])

    def __len__(self):
        return self._overlapped_frames_cum_sum[-1]

    def __getitem__(self, index: int) -> CorrespondencePair:
        if index > len(self):
            raise IndexError()

        img_1_index = bisect.bisect_right(self._overlapped_frames_cum_sum, index)

        if img_1_index > 0:
            img_2_index = (img_1_index + 1) + (index - self._overlapped_frames_cum_sum[img_1_index - 1])
        else:
            img_2_index = 1 + index

        img_sequence = self.img_sequence[img_1_index:img_2_index + 1]  # Add 1 since the end of a slice is exclusive

        pair_name = "{0}: {1} {2}".format(self.name, img_1_index, img_2_index)

        return KLTPair(img_sequence, self._tracker, pair_name)
