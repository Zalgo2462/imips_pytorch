from typing import Tuple

import numpy as np
import torch

import epipolar_nn.dataloaders.image
import epipolar_nn.dataloaders.pair


def generate_training_patches(
        pair: epipolar_nn.dataloaders.pair.StereoPair,
        img_1_keypoints_xy: torch.Tensor, img_2_keypoints_xy: torch.Tensor,
        patch_diameter: int, inlier_distance: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a training pair and the keypoints detected by the network for each image,
    return the training patches for the predicted anchors and true correspondences,
    as well as the inlier/outlier labels for each image.

    The patch diameter controls the size of the returned patches. This should
    correspond with the number of convolutions in the network. i.e.
    patch_diameter = num_conv_layers * 2 + 1 if each layer is a 3x3 convolution.

    Inlier_distance controls the labelling of inliers. The predicted anchor in one image
    and the true correspondence of the anchor predicted in the other image must be
    this close to be considered an inlier. If there exists a valid correspondence,
    but the distance from the anchor to the correspondence is beyond this limit,
    the prediction is labelled as an outlier.

    Returns an eight-tuple.
    The first four entries corresponds to the first image in the pair, and the last four entries
    corresponds to the second image.
    Each set has the form: (anchor patches, correspondence patches, inlier labels, outlier labels)
    Anchor patches and correspondence patches are tensors of the form BxCxHxW
    """
    # Convert keypoints to numpy data structures to compute the correspondences
    # The correspondences are calculated via numpy/ opencv, and therefore are likely
    # computed on the CPU
    img_1_keypoints_xy_np = img_1_keypoints_xy.cpu().numpy()
    img_2_keypoints_xy_np = img_2_keypoints_xy.cpu().numpy()

    img_1_packed_correspondences_xy, img_1_correspondences_indices = pair.correspondences(
        img_1_keypoints_xy_np,
        inverse=False
    )
    # img_1_correspondences_xy is (zero, zero).T where img_1_correspondences_mask is False
    # unpack_correspondences returns torch tensors, likely having the effect of moving the
    # data back to the GPU
    img_1_correspondences_xy, img_1_correspondences_mask = _unpack_correspondences(
        img_1_keypoints_xy_np,
        img_1_packed_correspondences_xy,
        img_1_correspondences_indices
    )

    img_2_packed_correspondences_xy, img_2_correspondences_indices = pair.correspondences(
        img_2_keypoints_xy_np,
        inverse=True
    )
    img_2_correspondences_xy, img_2_correspondences_mask = _unpack_correspondences(
        img_2_keypoints_xy_np,
        img_2_packed_correspondences_xy,
        img_2_correspondences_indices,
    )

    max_inlier_distance = torch.tensor([inlier_distance], device="cuda")

    # img_1_inlier_distances measures how far off the keypoints predicted in image 1
    # are from the true correspondences of the keypoints predicted from image 2
    img_1_inlier_distances = torch.norm((img_1_keypoints_xy - img_2_correspondences_xy), p=2, dim=0)
    img_1_inlier_distance_mask = (img_1_inlier_distances < max_inlier_distance)

    # inliers and outliers in this anchor image are determined by the true correspondences of the keypoints
    # detected when its paired image is ran through the net as an anchor image
    img_1_inliers = img_1_inlier_distance_mask & img_2_correspondences_mask
    img_1_outliers = ~img_1_inlier_distance_mask & img_2_correspondences_mask

    img_2_inlier_distances = torch.norm((img_2_keypoints_xy - img_1_correspondences_xy), p=2, dim=0)
    img_2_inlier_distance_mask = img_2_inlier_distances < max_inlier_distance

    img_2_inliers = img_2_inlier_distance_mask & img_1_correspondences_mask
    img_2_outliers = ~img_2_inlier_distance_mask & img_1_correspondences_mask

    img_1_kp_patches = _image_to_patch_batch(pair.image_1, img_1_keypoints_xy, patch_diameter)
    img_1_corr_patches = torch.zeros_like(img_1_kp_patches, device="cuda")
    img_1_corr_patches[img_2_correspondences_mask, :, :, :] = _image_to_patch_batch(
        pair.image_1,
        img_2_correspondences_xy[:, img_2_correspondences_mask],
        patch_diameter
    )
    img_2_kp_patches = _image_to_patch_batch(pair.image_2, img_2_keypoints_xy, patch_diameter)
    img_2_corr_patches = torch.zeros_like(img_2_kp_patches, device="cuda")
    img_2_corr_patches[img_1_correspondences_mask, :, :, :] = _image_to_patch_batch(
        pair.image_2,
        img_1_correspondences_xy[:, img_1_correspondences_mask],
        patch_diameter
    )

    # The results are returned in in two batches, one for each image that was subsampled.
    # This is a holdover from the original work. It might make more sense to
    # swap the kp and corr patches around in the future.
    # img 1_kp_patches come from the anchor points detected in img 1
    # img 1 corr patches come from the real correspondences of the anchor points detected in img 2
    # img 2 kp patches come from the anchor points detected in img 2
    # img 2 corr patches come from the real correspondences of the anchor points detected in img 1
    # img_1_inliers is true at an index when there is a valid, true correspondence in img_1 for the anchor in img_2
    #   and the anchor in img_1 is close to the true correspondence in img_1
    # img_2_inliers is true at an index when there is a valid, true correspondence in img_2 for the anchor in img_1
    #   and the anchor in img_2 is close to the true correspondence in img_2
    return img_1_kp_patches, img_1_corr_patches, img_1_inliers, img_1_outliers, \
           img_2_kp_patches, img_2_corr_patches, img_2_inliers, img_2_outliers


def _image_to_patch_batch(image_np: np.ndarray, keypoints_xy: torch.Tensor, diameter: int) -> torch.Tensor:
    """
    Extracts keypoints_xy.shape[1] patches from image_np centered on the given keypoints.
    Returns a tensor of diameter x diameter patches in a BxCxHxW tensor. The input image is
    expected to be a numpy image in either HxW or HxWxC format.
    """
    if diameter % 2 != 1:
        raise ValueError("diameter must be odd")

    keypoints_xy = keypoints_xy.to(torch.int)

    # this will copy the image from RAM to to the GPU,
    image = epipolar_nn.dataloaders.image.load_image_for_torch(image_np)
    radius = (diameter - 1) // 2

    batch = torch.zeros((keypoints_xy.shape[1], image.shape[0], diameter, diameter), device="cuda")
    for point_idx in range(keypoints_xy.shape[1]):
        keypoint_x = keypoints_xy[0, point_idx]
        keypoint_y = keypoints_xy[1, point_idx]
        batch[point_idx, :, :, :] = image[
                                    :,
                                    keypoint_y - radius: keypoint_y + radius + 1,
                                    keypoint_x - radius: keypoint_x + radius + 1
                                    ]
    return batch


def _unpack_correspondences(keypoints_xy: np.ndarray, correspondences_xy: np.ndarray, correspondence_indices) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the correspondences from their packed form to their unpacked form.

    In the packed form, the items of correspondences_xy are mapped to keypoints_xy
    by the correspondence_indices array. Where correspondences_xy[i] matches
    keypoints_xy[correspondence_indicies[i]].

    In the unpacked form, the correspondences are matched to keypoints_xy by index alone.
    If no correspondence exists for a given point in keypoints_xy, the resulting
    correspondence is (zero, zero).T. The correspondence_mask is aligned with
    keypoints_xy and unpacked_correspondences_xy such that correspondence_mask[i]
    is True when a valid correspondence exists for keypoints_xy[i] and False otherwise.

    This method takes in numpy arrays and returns torch tensors. This has the effect
    of moving the data from RAM into the GPU.
    """
    unpacked_correspondences_xy = np.zeros_like(keypoints_xy)
    # ValueError: shape mismatch: value array of shape (2,83) could not be broadcast to indexing result of shape (83,128)
    unpacked_correspondences_xy[:, correspondence_indices] = correspondences_xy

    correspondences_mask = np.zeros(keypoints_xy.shape[1], dtype=np.bool)
    correspondences_mask[correspondence_indices] = True

    return torch.tensor(unpacked_correspondences_xy, device="cuda"), torch.tensor(correspondences_mask, device="cuda")
