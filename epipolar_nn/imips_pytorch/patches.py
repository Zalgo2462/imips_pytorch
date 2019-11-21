from typing import Tuple

import numpy as np
import torch

import epipolar_nn.dataloaders.pair


def generate_training_patches(
        pair: epipolar_nn.dataloaders.pair.StereoPair,
        img_1_keypoints_xy: torch.Tensor, img_2_keypoints_xy: torch.Tensor,
        patch_diameter: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Grab numpy references to the keypoint data.
    # TODO: figure out how this conflicts with gpu/ cuda

    # Convert keypoints to numpy data structures to compute the correspondences
    img_1_keypoints_xy_np = img_1_keypoints_xy.numpy()
    img_2_keypoints_xy_np = img_2_keypoints_xy.numpy()

    img_1_packed_correspondences_xy, img_1_correspondences_indices = pair.correspondences(
        img_1_keypoints_xy.numpy(),
        inverse=False
    )
    # img_1_correspondences_xy is (zero, zero).T where img_1_correspondences_mask is False
    img_1_correspondences_xy, img_1_correspondences_mask = _unpack_correspondences(
        img_1_keypoints_xy_np,
        img_1_packed_correspondences_xy,
        img_1_correspondences_indices
    )

    img_2_packed_correspondences_xy, img_2_correspondences_indices = pair.correspondences(
        img_2_keypoints_xy.numpy(),
        inverse=True
    )
    img_2_correspondences_xy, img_2_correspondences_mask = _unpack_correspondences(
        img_2_keypoints_xy_np,
        img_2_packed_correspondences_xy,
        img_2_correspondences_indices,
    )

    max_inlier_distance = torch.tensor([3])

    # img_1_inlier_distances measures how far off the keypoints predicted in image 1
    # are from the correspondences of the keypoints predicted from image 2
    img_1_inlier_distances = torch.norm((img_1_keypoints_xy - img_2_correspondences_xy), p=2, dim=0)
    img_1_inlier_distance_mask = (img_1_inlier_distances < torch.tensor(max_inlier_distance))

    # inliers and outliers in this anchor image are determined by the true correspondences of the keypoints
    # detected when its paired image is ran through the net as an anchor image
    img_1_inliers = img_1_inlier_distance_mask & img_2_correspondences_mask
    img_1_outliers = ~img_1_inlier_distance_mask & img_2_correspondences_mask

    img_2_inlier_distances = torch.norm((img_2_keypoints_xy - img_1_correspondences_xy), p=2, dim=0)
    img_2_inlier_distance_mask = img_2_inlier_distances < max_inlier_distance

    img_2_inliers = img_2_inlier_distance_mask & img_1_correspondences_mask
    img_2_outliers = img_2_inlier_distance_mask & img_1_correspondences_mask

    img_1_kp_patches = _image_to_patch_batch(pair.image_1, img_1_keypoints_xy, patch_diameter)
    img_1_corr_patches = torch.zeros_like(img_1_kp_patches)
    img_1_corr_patches[img_2_correspondences_mask, :, :, :] = _image_to_patch_batch(
        pair.image_1,
        img_2_correspondences_xy[img_2_correspondences_mask],
        patch_diameter
    )
    img_2_kp_patches = _image_to_patch_batch(pair.image_2, img_2_keypoints_xy, patch_diameter)
    img_2_corr_patches = torch.zeros_like(img_2_kp_patches)
    img_2_corr_patches[img_1_correspondences_mask, :, :, :] = _image_to_patch_batch(
        pair.image_2,
        img_1_correspondences_xy[img_1_correspondences_mask],
        patch_diameter
    )

    # The results are returned in in two batches, one for each image that was subsampled.
    # This is a holdover from the original work. It might make more sense to
    # swap the kp and corr patches around in the future.
    # img 1_kp_patches come from the anchor points detected in img 1
    # img 2 corr patches come from the real correspondences of the anchor points detected in img 1
    # img 2 kp patches come from the anchor points detected in img 2
    # img 1 corr patches come from the real correspondences of the anchor points detected in img 2

    return (
        (img_1_kp_patches, img_1_corr_patches, img_1_inliers, img_1_outliers),
        (img_2_kp_patches, img_2_corr_patches, img_2_inliers, img_2_outliers),
    )


def _image_to_patch_batch(image_np: np.ndarray, keypoints_xy: torch.Tensor, diameter: int) -> torch.Tensor:
    if diameter % 2 != 1:
        raise ValueError("diameter must be odd")

    image = torch.tensor(image_np)
    radius = (diameter - 1) / 2

    multichannel = len(image.shape) == 3

    if multichannel:
        batch = torch.zeros((keypoints_xy.shape[1], diameter, diameter, image.shape[2]))
    else:
        batch = torch.zeros((keypoints_xy.shape[1], diameter, diameter, 1))

    for point_idx in range(keypoints_xy.shape[1]):
        keypoint_x = keypoints_xy[0, point_idx]
        keypoint_y = keypoints_xy[1, point_idx]
        if multichannel:
            batch[point_idx, :, :, :] = image[
                                        keypoint_x - radius: keypoint_x + radius,
                                        keypoint_y - radius: keypoint_y + radius,
                                        :,
                                        ]
        else:
            batch[point_idx, :, :, 0] = image[
                                        keypoint_x - radius: keypoint_x + radius,
                                        keypoint_y - radius: keypoint_y + radius,
                                        ]
    return batch


def _unpack_correspondences(keypoints_xy: np.ndarray, correspondences_xy: np.ndarray, correspondence_indices) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    unpacked_correspondences_xy = np.zeros_like(keypoints_xy)
    unpacked_correspondences_xy[correspondence_indices] = correspondences_xy

    correspondences_mask = np.zeros(keypoints_xy.shape[1], dtype=np.bool)
    correspondences_mask[correspondence_indices] = True

    return torch.tensor(unpacked_correspondences_xy), torch.tensor(correspondences_mask)
