from abc import ABC
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch

from .image import load_image_for_torch


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

    @staticmethod
    def collate_for_torch(pairs: List['ImagePair']):
        image_1_tensors = [load_image_for_torch(pair.image_1) for pair in pairs]
        image_2_tensors = [load_image_for_torch(pair.image_2) for pair in pairs]
        names = [pair.name for pair in pairs]

        return torch.stack(image_1_tensors), torch.stack(image_2_tensors), names


class CorrespondencePair(ImagePair, ABC):

    def correspondences(self, pixels_xy: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @staticmethod
    def collate_for_torch(pairs: List['CorrespondencePair']):
        image_1_tensors, image_2_tensors, names = ImagePair.collate_for_torch(pairs)
        # Batch up the correspondence functions for each pair, this likely closes over the
        # original numpy images
        correspondence_funcs = [pair.correspondences for pair in pairs]
        return image_1_tensors, image_2_tensors, names, correspondence_funcs


class FundamentalMatrixPair(ImagePair, ABC):

    @property
    def f_matrix_forward(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def f_matrix_backward(self) -> np.ndarray:
        raise NotImplementedError()

    def generate_virtual_points(self, step: Optional[float] = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        # set grid points for each image
        grid_x, grid_y = np.meshgrid(
            np.arange(0, 1, step), np.arange(0, 1, step)
        )
        num_points_eval = len(grid_x.flatten())

        pts1_grid = np.float32(
            np.vstack(
                (self.image_1.shape[1] * grid_x.flatten(), self.image_1.shape[0] * grid_y.flatten())
            ).T
        )[np.newaxis, :, :]
        pts2_grid = np.float32(
            np.vstack(
                (self.image_2.shape[1] * grid_x.flatten(), self.image_2.shape[1] * grid_y.flatten())
            ).T
        )[np.newaxis, :, :]

        pts1_virt, pts2_virt = cv2.correctMatches(self.f_matrix_forward, pts1_grid, pts2_grid)

        valid_1 = np.logical_and(
            np.logical_not(np.isnan(pts1_virt[:, :, 0])),
            np.logical_not(np.isnan(pts1_virt[:, :, 1])),
        )
        valid_2 = np.logical_and(
            np.logical_not(np.isnan(pts2_virt[:, :, 0])),
            np.logical_not(np.isnan(pts2_virt[:, :, 1])),
        )

        _, valid_idx = np.where(np.logical_and(valid_1, valid_2))
        good_pts = len(valid_idx)

        while good_pts < num_points_eval:
            valid_idx = np.hstack(
                (valid_idx, valid_idx[: (num_points_eval - good_pts)])
            )
            good_pts = len(valid_idx)

        valid_idx = valid_idx[: num_points_eval]

        pts1_virt = pts1_virt[:, valid_idx]
        pts2_virt = pts2_virt[:, valid_idx]

        ones = np.ones((pts1_virt.shape[1], 1))

        pts1_virt = np.hstack((pts1_virt[0], ones))
        pts2_virt = np.hstack((pts2_virt[0], ones))
        return pts1_virt, pts2_virt

    @staticmethod
    def collate_for_torch(pairs: List['FundamentalMatrixPair']):
        image_1_tensors, image_2_tensors, names = ImagePair.collate_for_torch(pairs)
        f_mats_forward = [torch.tensor(pair.f_matrix_forward, dtype=torch.float32) for pair in pairs]
        f_mats_backward = [torch.tensor(pair.f_matrix_backward, dtype=torch.float32) for pair in pairs]
        virt_pts = [pair.generate_virtual_points() for pair in pairs]
        pts_1_virt = [torch.tensor(virt_pt[0], dtype=torch.float32) for virt_pt in virt_pts]
        pts_2_virt = [torch.tensor(virt_pt[1], dtype=torch.float32) for virt_pt in virt_pts]
        return image_1_tensors, image_2_tensors, names, torch.stack(f_mats_forward), torch.stack(f_mats_backward), \
               torch.stack(pts_1_virt), torch.stack(pts_2_virt)


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

    def collate_for_torch(pairs: List['CorrespondenceFundamentalMatrixPair']):
        image_1_tensors, image_2_tensors, names = ImagePair.collate_for_torch(pairs)
        # Batch up the correspondence functions for each pair, this likely closes over the
        # original numpy images
        correspondence_funcs = [pair.correspondences for pair in pairs]

        f_mats_forward = [torch.tensor(pair.f_matrix_forward, dtype=torch.float32) for pair in pairs]
        f_mats_backward = [torch.tensor(pair.f_matrix_backward, dtype=torch.float32) for pair in pairs]
        virt_pts = [pair.generate_virtual_points() for pair in pairs]
        pts_1_virt = [torch.tensor(virt_pt[0], dtype=torch.float32) for virt_pt in virt_pts]
        pts_2_virt = [torch.tensor(virt_pt[1], dtype=torch.float32) for virt_pt in virt_pts]
        return image_1_tensors, image_2_tensors, correspondence_funcs, names, \
               torch.stack(f_mats_forward), torch.stack(f_mats_backward), \
               torch.stack(pts_1_virt), torch.stack(pts_2_virt)
