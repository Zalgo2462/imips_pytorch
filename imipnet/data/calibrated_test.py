import unittest

import numpy as np
from scipy.stats import special_ortho_group

from imipnet.data.calibrated import PinvFundamentalMatrixPair, StdStereoFundamentalMatrixPair


class TestFundamentalMatrixPairs(unittest.TestCase):

    def _generate_random_camera_data(self, n):
        random_intrinsic_matrices = np.random.rand(n, 3, 3)
        random_intrinsic_matrices[:, 0, 1] = 0
        random_intrinsic_matrices[:, 1, 0] = 0
        random_intrinsic_matrices[:, 2, :] = np.array([0, 0, 1])
        random_intrinsic_matrices *= 1000

        random_intrinsic_matrices_inv = np.linalg.inv(random_intrinsic_matrices)

        random_extrinsic_matrices = np.random.rand(n, 3, 4) * 2
        random_translation_vectors = random_extrinsic_matrices[:, :, 3:4]
        random_rotation_matrices = special_ortho_group.rvs(dim=3, size=500)
        random_extrinsic_matrices[:, :, 0:3] = random_rotation_matrices

        random_pose_matrices = np.block([
            [random_rotation_matrices.transpose((0, 2, 1)),
             -1 * random_rotation_matrices.transpose((0, 2, 1)) @ random_translation_vectors]
        ])
        random_camera_centers = random_pose_matrices[:, :, 3:4]

        return random_intrinsic_matrices, random_intrinsic_matrices_inv, random_extrinsic_matrices, random_pose_matrices, random_camera_centers

    def test_std_stereo_pinv_epipoles(self):
        n = 500
        epsilon = 1e-6

        random_intrinsic_matrices, random_intrinsic_matrices_inv, random_extrinsic_matrices, random_pose_matrices, random_camera_centers = self._generate_random_camera_data(
            n)
        pinv_pairs = []
        std_stereo_pairs = []
        image_1_epipoles = []
        image_2_epipoles = []

        for i in range(n):
            id1 = i
            id2 = (i + 1) % n
            image_1_camera_center = random_camera_centers[id1, :, :]
            image_2_camera_center = random_camera_centers[id2, :, :]
            image_1_intrinsic_matrix = random_intrinsic_matrices[id1, :, :]
            image_2_intrinsic_matrix = random_intrinsic_matrices[id2, :, :]
            image_1_intrinsic_matrix_inv = random_intrinsic_matrices_inv[id1, :, :]
            image_2_intrinsic_matrix_inv = random_intrinsic_matrices_inv[id2, :, :]
            image_1_camera_matrix = image_1_intrinsic_matrix @ random_extrinsic_matrices[id1, :, :]
            image_2_camera_matrix = image_2_intrinsic_matrix @ random_extrinsic_matrices[id2, :, :]
            image_1_pose_matrix = random_pose_matrices[id1, :, :]
            image_2_pose_matrix = random_pose_matrices[id2, :, :]
            image_1_extrinsic_matrix = random_extrinsic_matrices[id1, :, :]
            image_2_extrinsic_matrix = random_extrinsic_matrices[id2, :, :]

            image_1_epipole = image_1_camera_matrix @ np.vstack((image_2_camera_center, 1))
            image_1_epipole = image_1_epipole / image_1_epipole[-1, 0]
            image_2_epipole = image_2_camera_matrix @ np.vstack((image_1_camera_center, 1))
            image_2_epipole = image_2_epipole / image_2_epipole[-1, 0]

            image_1_epipoles.append(image_1_epipole.squeeze())
            image_2_epipoles.append(image_2_epipole.squeeze())

            pinv_pairs.append(
                PinvFundamentalMatrixPair(
                    np.array([]), np.array([]), "",
                    image_1_camera_center, image_2_camera_center,
                    image_1_camera_matrix, image_2_camera_matrix
                )
            )

            std_stereo_pairs.append(
                StdStereoFundamentalMatrixPair(
                    np.array([]), np.array([]), "",
                    image_1_intrinsic_matrix, image_2_intrinsic_matrix,
                    image_1_intrinsic_matrix_inv, image_2_intrinsic_matrix_inv,
                    image_1_extrinsic_matrix, image_2_extrinsic_matrix,
                    image_1_pose_matrix, image_2_pose_matrix
                )
            )

        for pinv_pair, std_stereo_pair, image_1_epipole, image_2_epipole in zip(pinv_pairs, std_stereo_pairs,
                                                                                image_1_epipoles, image_2_epipoles):
            pinv_f_matrix = pinv_pair.f_matrix_forward
            std_stereo_f_matrix = std_stereo_pair.f_matrix_forward

            pinv_f_matrix = pinv_f_matrix / np.linalg.norm(pinv_f_matrix)
            std_stereo_f_matrix = std_stereo_f_matrix / np.linalg.norm(std_stereo_f_matrix)

            # Ensure the std_stereo f matrix matches the pinv f matrix
            self.assertLess(np.sum(np.abs(pinv_f_matrix - std_stereo_f_matrix), axis=(0, 1)), epsilon)

            # Ensure epipoles match the projections of the opposite camera centers
            u, _, v = np.linalg.svd(pinv_f_matrix)
            epi_1 = v.T[:, -1]
            epi_1 = epi_1 / epi_1[-1]
            epi_2 = u[:, -1]
            epi_2 = epi_2 / epi_2[-1]

            self.assertLess(np.sum(np.abs(epi_1 - image_1_epipole)), epsilon)
            self.assertLess(np.sum(np.abs(epi_2 - image_2_epipole)), epsilon)
