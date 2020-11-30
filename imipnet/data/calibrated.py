import numpy as np

from imipnet.data.pairs import FundamentalMatrixPair

PINV_F_MAT_ALGORITHM = 0
STD_STEREO_F_MAT_ALGORITHM = 1


class PinvFundamentalMatrixPair(FundamentalMatrixPair):
    def __init__(self, image_1: np.ndarray, image_2: np.ndarray, name: str,
                 image_1_camera_center: np.ndarray, image_2_camera_center: np.ndarray,
                 image_1_camera_matrix: np.ndarray, image_2_camera_matrix: np.ndarray):
        self._image_1 = image_1
        self._image_2 = image_2
        self._name = name

        self._F_mat_forward = PinvFundamentalMatrixPair.calc_f_matrix(
            image_1_camera_center, image_1_camera_matrix, image_2_camera_matrix
        )
        self._F_mat_backward = PinvFundamentalMatrixPair.calc_f_matrix(
            image_2_camera_center, image_2_camera_matrix, image_1_camera_matrix
        )

    @property
    def f_matrix_forward(self) -> np.ndarray:
        return self._F_mat_forward

    @property
    def f_matrix_backward(self) -> np.ndarray:
        return self._F_mat_backward

    @property
    def image_1(self) -> np.ndarray:
        return self._image_1

    @property
    def image_2(self) -> np.ndarray:
        return self._image_2

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def calc_f_matrix(image_1_camera_center: np.ndarray, image_1_camera_matrix: np.ndarray,
                      image_2_camera_matrix: np.ndarray) -> np.ndarray:
        # Calculate the epipole in image 2
        epipole = image_2_camera_matrix @ np.vstack((image_1_camera_center, 1))
        epipole_cross_mat = _cross_mat(epipole)
        f = epipole_cross_mat @ image_2_camera_matrix @ np.linalg.pinv(image_1_camera_matrix)
        return f


class StdStereoFundamentalMatrixPair(FundamentalMatrixPair):

    def __init__(self, image_1: np.ndarray, image_2: np.ndarray, name: str,
                 image_1_intrinsic_matrix: np.ndarray, image_2_intrinsic_matrix: np.ndarray,
                 image_1_intrinsic_matrix_inv: np.ndarray, image_2_intrinsic_matrix_inv: np.ndarray,
                 image_1_extrinsic_matrix: np.ndarray, image_2_extrinsic_matrix: np.ndarray,
                 image_1_pose_matrix: np.ndarray, image_2_pose_matrix: np.ndarray
                 ):
        self._image_1 = image_1
        self._image_2 = image_2
        self._name = name

        self.baseline_1_2 = image_1_extrinsic_matrix @ np.vstack((image_2_pose_matrix[:, -1, np.newaxis], 1))
        self.rotation_rad = np.arccos((np.trace(image_1_pose_matrix[:, 0:3] @ image_2_pose_matrix[:, 0:3].T) - 1) / 2)

        self._F_mat_forward = StdStereoFundamentalMatrixPair.calc_f_matrix(
            image_1_intrinsic_matrix_inv, image_1_pose_matrix, image_2_intrinsic_matrix, image_2_extrinsic_matrix
        )
        self._F_mat_backward = StdStereoFundamentalMatrixPair.calc_f_matrix(
            image_2_intrinsic_matrix_inv, image_2_pose_matrix, image_1_intrinsic_matrix, image_1_extrinsic_matrix
        )

    @property
    def f_matrix_forward(self) -> np.ndarray:
        return self._F_mat_forward

    @property
    def f_matrix_backward(self) -> np.ndarray:
        return self._F_mat_backward

    @property
    def image_1(self) -> np.ndarray:
        return self._image_1

    @property
    def image_2(self) -> np.ndarray:
        return self._image_2

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def calc_f_matrix(image_1_intrinsic_matrix_inverse: np.ndarray, image_1_pose_matrix: np.ndarray,
                      image_2_intrinsic_matrix: np.ndarray,
                      image_2_extrinsic_matrix: np.ndarray) -> np.ndarray:
        # Shifts the world frame to the reference frame of image 1.
        # Camera 1's External matrix becomes [I | 0].
        # Then computes the fundamental matrix as [K't]_x K' R K^{−1}
        # See Hartley and Zisserman p. 244

        extrinsic_matrix = np.vstack((image_2_extrinsic_matrix, [0, 0, 0, 1])) @ np.vstack(
            (image_1_pose_matrix, [0, 0, 0, 1]))
        rotation_matrix = extrinsic_matrix[0:3, 0:3]
        translation_vec = extrinsic_matrix[0:3, 3:4]
        epipole = image_2_intrinsic_matrix @ translation_vec
        epipole_cross_mat = _cross_mat(epipole)
        return epipole_cross_mat @ image_2_intrinsic_matrix @ rotation_matrix @ image_1_intrinsic_matrix_inverse


def _cross_mat(column_vec3: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -column_vec3[2][0], column_vec3[1][0]],
        [column_vec3[2][0], 0, -column_vec3[0][0]],
        [-column_vec3[1][0], column_vec3[0][0], 0]
    ])
