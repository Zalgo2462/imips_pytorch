import unittest

import numpy as np
import torch

from .trainer import ImipTrainer


class TestImipsTrainer(unittest.TestCase):
    def test_unpack_correspondences(self):
        anchors = np.random.randint(0, 640, (2, 500))
        correspondence_mask = np.random.rand(500) > .5
        correspondence_indices = np.arange(500)[correspondence_mask]

        unpacked_correspondences = np.zeros_like(anchors)
        unpacked_correspondences[:, correspondence_mask] = anchors[:, correspondence_mask]
        packed_correspondences = anchors[:, correspondence_indices]

        test_unpacked_correspondences, test_correspondence_mask = ImipTrainer._unpack_correspondences(anchors,
                                                                                                      packed_correspondences,
                                                                                                      correspondence_indices)
        test_unpacked_correspondences = test_unpacked_correspondences.cpu().numpy()
        test_correspondence_mask = test_correspondence_mask.cpu().numpy()

        self.assertTrue(np.array_equal(unpacked_correspondences, test_unpacked_correspondences))
        self.assertTrue(np.array_equal(correspondence_mask, test_correspondence_mask))

    def test_label_inliers_outliers(self):
        inlier_radius = 3

        img_1_keypoints_xy = np.random.randint(0, 640, (2, 500))  # keypoints in image 1
        img_2_keypoints_xy = np.random.randint(0, 640, (2, 500))  # keypoints in image 2

        img_1_corr_offsets = np.random.randint(0, 10, (2, 500))  # offsets from img_2_keypoints for img_1_corrs
        img_2_corr_offsets = np.random.randint(0, 10, (2, 500))  # offsets from img_1_keypoints for img_2_corrs

        img_1_correspondences_xy = img_2_keypoints_xy + img_1_corr_offsets
        img_2_correspondences_xy = img_1_keypoints_xy + img_2_corr_offsets

        img_1_inliers = np.linalg.norm(img_2_corr_offsets, axis=0) < inlier_radius
        img_2_inliers = np.linalg.norm(img_1_corr_offsets, axis=0) < inlier_radius

        img_1_corr_mask = np.random.rand(500) > .5
        img_2_corr_mask = np.random.rand(500) > .5

        img_1_correspondences_xy[:, ~img_1_corr_mask] = np.zeros((2, 1))
        img_2_correspondences_xy[:, ~img_2_corr_mask] = np.zeros((2, 1))

        img_1_outliers = ~img_1_inliers & img_2_corr_mask
        img_2_outliers = ~img_2_inliers & img_1_corr_mask
        img_1_inliers = img_1_inliers & img_2_corr_mask
        img_2_inliers = img_2_inliers & img_1_corr_mask

        test_img_1_inliers, test_img_1_outliers, test_img_2_inliers, test_img_2_outliers = ImipTrainer.label_inliers_outliers(
            torch.tensor(img_1_keypoints_xy, device="cuda", dtype=torch.float32),
            torch.tensor(img_1_correspondences_xy, device="cuda", dtype=torch.float32),
            torch.tensor(img_1_corr_mask, device="cuda"),
            torch.tensor(img_2_keypoints_xy, device="cuda", dtype=torch.float32),
            torch.tensor(img_2_correspondences_xy, device="cuda", dtype=torch.float32),
            torch.tensor(img_2_corr_mask, device="cuda"),
            inlier_radius
        )

        self.assertTrue(np.array_equal(img_1_inliers, test_img_1_inliers.cpu().numpy()))
        self.assertTrue(np.array_equal(img_1_outliers, test_img_1_outliers.cpu().numpy()))
        self.assertTrue(np.array_equal(img_2_inliers, test_img_2_inliers.cpu().numpy()))
        self.assertTrue(np.array_equal(img_2_outliers, test_img_2_outliers.cpu().numpy()))

    def test_image_to_patch_batch(self):
        image = np.random.randint(0, 255, (480, 640, 1))
        diameter = 15
        radius = (diameter - 1) // 2
        keypoints_xy = np.random.randint(0 + radius, 480 - radius, (2, 128))

        patch_batch = np.zeros((128, 1, diameter, diameter))
        for i in range(128):
            kp_x = keypoints_xy[0, i]
            kp_y = keypoints_xy[1, i]
            patch_batch[i, 0, :, :] = image[kp_y - radius: kp_y + radius + 1, kp_x - radius: kp_x + radius + 1, 0]

        test_patch_batch = ImipTrainer.image_to_patch_batch(image, torch.tensor(keypoints_xy, device="cuda"),
                                                            diameter)
        test_patch_batch = test_patch_batch.cpu().numpy()

        self.assertTrue(np.array_equal(test_patch_batch, patch_batch))
