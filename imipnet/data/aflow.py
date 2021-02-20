from typing import Tuple

import numpy as np

from imipnet.data.pairs import CorrespondencePair


class AbsoluteFlowPair(CorrespondencePair):
    """
    AbsoluteFlowPair provides stereo correspondences by way of two 2xHxW arrays
    which contain the absolute, forward and backward optical flow. This makes
    finding correspondences as simple as indexing the arrays.
    NaN should be used to mark locations for which there is no flow data.
    """

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


class CalibratedDepthPair(AbsoluteFlowPair):
    """
    CalibratedDepthPair provides stereo correspondences by constructing
    absolute flow maps between two images from their pinhole camera calibration
    and their depth maps. NaN should be used to mark locations for which
    there is no depth data.
    """

    def __init__(self, image_1: np.ndarray, image_2: np.ndarray, name: str,
                 image_1_camera_matrix: np.ndarray, image_2_camera_matrix: np.ndarray,
                 image_1_depth_map: np.ndarray, image_2_depth_map: np.ndarray
                 ):
        image_1_shape = image_1.shape[:2]
        image_2_shape = image_2.shape[:2]
        abs_flow_forward = CalibratedDepthPair._calculate_absolute_flow(
            image_1_camera_matrix, image_2_camera_matrix, image_1_depth_map,
            image_1_shape, image_2_shape
        )
        abs_flow_backward = CalibratedDepthPair._calculate_absolute_flow(
            image_2_camera_matrix, image_1_camera_matrix, image_2_depth_map,
            image_2_shape, image_1_shape
        )
        abs_flow_forward, abs_flow_backward = CalibratedDepthPair._refine_pairwise_absolute_flow(
            abs_flow_forward, abs_flow_backward,
        )
        super(CalibratedDepthPair, self).__init__(
            image_1, image_2, name, abs_flow_forward, abs_flow_backward
        )

    @staticmethod
    def _calculate_absolute_flow(image_1_camera_matrix: np.ndarray, image_2_camera_matrix: np.ndarray,
                                 image_1_depth_map: np.ndarray, image_1_shape: Tuple[int, int],
                                 image_2_shape: Tuple[int, int]) -> np.ndarray:
        """

        :param image_1_camera_matrix: 3x4 pinhole camera matrix for image 1
        :param image_2_camera_matrix: 3x4 pinhole camera matrix for image 2
        :param image_1_depth_map: depth map for image 1 with units matching camera matrices.
               NaN should be used to mark locations for which there is no depth data.
        :param image_1_shape: shape of image 1, must match shape of the provided depth map
        :param image_2_shape: shape of image 2
        :return: forward absolute flow from image 1 to image 2
        """
        # create inverse projection ala
        # https://github.com/colmap/colmap/blob/ff9a463067a2656d1f59d12109fe2931e29e3ca0/src/mvs/image.cc#L115
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/mvs/fusion.cc#L373
        image_1_camera_matrix_inv = np.linalg.inv(np.block([[image_1_camera_matrix], [np.array([0, 0, 0, 1])]]))

        # camera_inv @ [col * depth, row * depth, depth, 1].T
        x, y = np.meshgrid(np.arange(image_1_shape[1], dtype=np.float32),
                           np.arange(image_1_shape[0], dtype=np.float32))
        ones = np.ones_like(x)
        xy = np.stack((x, y, ones, ones), axis=0)
        xy[:3, :, :] = xy[:3, :, :] * np.expand_dims(image_1_depth_map, axis=0)

        abs_flow = (image_2_camera_matrix @ (image_1_camera_matrix_inv @ xy.reshape(4, -1)))
        abs_flow = abs_flow[0:2] / abs_flow[2:3]

        with np.errstate(invalid='ignore'):  # silence warnings stemming from nan comparisons
            bad_flow = ((abs_flow[0, :] < 0) | (abs_flow[1, :] < 0) |
                        (abs_flow[0, :] >= image_2_shape[1]) | (abs_flow[1, :] >= image_2_shape[0]))

        bad_flow = np.stack((bad_flow, bad_flow), axis=0)
        abs_flow[bad_flow] = float('nan')

        abs_flow = abs_flow.reshape(2, image_1_shape[0], image_1_shape[1])
        return abs_flow.astype(np.float32)

    @staticmethod
    def _refine_pairwise_absolute_flow(image_1_abs_flow: np.ndarray, image_2_abs_flow: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        _refine_pairwise_absolute_flow ensures that the resulting flow has forward-backward consistency

        :param image_1_abs_flow: absolute flow from image 1 to image 2 ([x,y]xHxW)
        :param image_2_abs_flow: absolute flow from image 2 to image 1 ([x,y]xHxW)
        :return: Updated versions of the input flows with the inconsistent entries NaN'd out
        """
        image_1_flow_shape = image_1_abs_flow.shape
        image_2_flow_shape = image_2_abs_flow.shape

        # shift to a linear perspective to make the indexing easier
        image_1_flow_linear = image_1_abs_flow.reshape(2, -1)
        image_2_flow_linear = image_2_abs_flow.reshape(2, -1)

        # remove the nans, but keep the index so we can remake the flow
        image_1_non_nan_index = np.arange(image_1_flow_linear.shape[1])[~np.isnan(image_1_flow_linear[0])]
        image_1_flow_filtered = image_1_flow_linear[:, image_1_non_nan_index]

        image_2_non_nan_index = np.arange(image_2_flow_linear.shape[1])[~np.isnan(image_2_flow_linear[0])]
        image_2_flow_filtered = image_2_flow_linear[:, image_2_non_nan_index]

        # round accounting for border pixels
        image_1_flow_filtered = np.round(image_1_flow_filtered).astype(np.int)
        image_1_flow_filtered[0, image_1_flow_filtered[0] == image_2_flow_shape[2]] -= 1
        image_1_flow_filtered[1, image_1_flow_filtered[1] == image_2_flow_shape[1]] -= 1

        image_2_flow_filtered = np.round(image_2_flow_filtered).astype(np.int)
        image_2_flow_filtered[0, image_2_flow_filtered[0] == image_1_flow_shape[2]] -= 1
        image_2_flow_filtered[1, image_2_flow_filtered[1] == image_1_flow_shape[1]] -= 1

        # check if indexing the other flow field with the first field results in a nan flow
        image_1_bad_reverse_flow = np.isnan(
            image_2_abs_flow[0, image_1_flow_filtered[1], image_1_flow_filtered[0]]
        )
        image_1_flow_linear[:, image_1_non_nan_index[image_1_bad_reverse_flow]] = float('nan')

        image_2_bad_reverse_flow = np.isnan(
            image_1_abs_flow[0, image_2_flow_filtered[1], image_2_flow_filtered[0]]
        )
        image_2_flow_linear[:, image_2_non_nan_index[image_2_bad_reverse_flow]] = float('nan')

        image_1_abs_flow = image_1_flow_linear.reshape(image_1_flow_shape)
        image_2_abs_flow = image_2_flow_linear.reshape(image_2_flow_shape)

        return image_1_abs_flow, image_2_abs_flow
