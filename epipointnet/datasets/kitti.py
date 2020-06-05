import bisect
import os
import pickle
import shutil
from typing import Optional, Dict, Tuple

import numpy as np
import torch.utils.data
import torchvision.datasets.utils as tv_data

from epipointnet.data import pairs, klt, calibrated
from epipointnet.datasets import sequence


class KITTIMonocularStereoPairsSequence(torch.utils.data.Dataset):
    url_color = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip"
    url_gray = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"
    url_pose = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
    url_calibration_map = {
        '00': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip',
        '01': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip',
        '02': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip',
        '03': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip',
        '04': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
        '05': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
        '06': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
        '07': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
        '08': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
        '09': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
        '10': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip',
    }
    _file_calibration_map = {
        '00': os.path.join('calibration', "2011_10_03", "calib_cam_to_cam.txt"),
        '01': os.path.join('calibration', "2011_10_03", "calib_cam_to_cam.txt"),
        '02': os.path.join('calibration', "2011_10_03", "calib_cam_to_cam.txt"),
        '03': os.path.join('calibration', "2011_09_26", "calib_cam_to_cam.txt"),
        '04': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
        '05': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
        '06': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
        '07': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
        '08': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
        '09': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
        '10': os.path.join('calibration', "2011_09_30", "calib_cam_to_cam.txt"),
    }

    @property
    def _raw_folder(self: 'KITTIMonocularStereoPairsSequence') -> str:
        return os.path.join(self._root_folder, self.__class__.__name__, "color" if self._color else "gray", 'raw')

    @property
    def _processed_folder(self: 'KITTIMonocularStereoPairsSequence') -> str:
        return os.path.join(self._root_folder, self.__class__.__name__, "color" if self._color else "gray", 'processed')

    @property
    def _processed_sequence_folder(self: 'KITTIMonocularStereoPairsSequence') -> str:
        return os.path.join(self._processed_folder, self._sequence)

    def __init__(self: 'KITTIMonocularStereoPairsSequence',
                 root: str,
                 kitti_sequence: str,
                 download: Optional[bool] = True,
                 color: Optional[bool] = True,
                 minimum_KLT_overlap: Optional[float] = 0.3,
                 f_matrix_algorithm: Optional[int] = None) -> None:
        self._root_folder = os.path.abspath(root)
        self._sequence = kitti_sequence
        self._color = color

        self._tracker = klt.Tracker()

        if download:
            self.download()

        # load the camera calibration data
        with open(os.path.join(self._processed_sequence_folder, "calibration_data.pickle"),
                  'rb') as calib_data_file_obj:
            calibration_data = pickle.load(calib_data_file_obj)
        self._intrinsic_matrix = calibration_data[0]
        self._intrinsic_matrix_inv = np.linalg.inv(self._intrinsic_matrix)
        self._pose_matrices = calibration_data[1]
        self._extrinsic_matrices = calibration_data[2]

        # load the images
        img_path = os.path.join(self._processed_sequence_folder, "images")
        self._image_sequence = sequence.GlobImageSequence(os.path.join(img_path, "*.png"),
                                                          convert_to_grayscale=not color)

        # If the image data doesn't match the pose data, quit.
        assert len(self._image_sequence) == self._pose_matrices.shape[0]

        # load the klt data
        with open(os.path.join(self._processed_sequence_folder, "overlap.pickle"), 'rb') as overlap_file:
            seq_overlap = pickle.load(overlap_file)

        self._overlapped_frames_cum_sum = []
        overlapped_frames = seq_overlap.find_frames_with_overlap(0, minimum_KLT_overlap).size
        self._overlapped_frames_cum_sum.append(overlapped_frames)
        for i in range(1, len(self._image_sequence) - 1):
            overlapped_frames = seq_overlap.find_frames_with_overlap(i, minimum_KLT_overlap).size
            self._overlapped_frames_cum_sum.append(overlapped_frames + self._overlapped_frames_cum_sum[-1])

        if f_matrix_algorithm is None:
            f_matrix_algorithm = calibrated.PINV_F_MAT_ALGORITHM

        assert f_matrix_algorithm in [calibrated.PINV_F_MAT_ALGORITHM, calibrated.STD_STEREO_F_MAT_ALGORITHM]

        self._f_matrix_algorithm = f_matrix_algorithm

    def __len__(self: 'KITTIMonocularStereoPairsSequence') -> int:
        return self._overlapped_frames_cum_sum[-1]

    def __getitem__(self, index: int):
        if index > len(self):
            raise IndexError()

        img_1_index = bisect.bisect_right(self._overlapped_frames_cum_sum, index)

        if img_1_index > 0:
            img_2_index = (img_1_index + 1) + (index - self._overlapped_frames_cum_sum[img_1_index - 1])
        else:
            img_2_index = 1 + index

        img_sequence = self._image_sequence[img_1_index:img_2_index + 1]  # Add 1 since the end of a slice is exclusive
        pair_name = "KITTI ODOM {0}: {1} {2}".format(self._sequence, img_1_index, img_2_index)

        klt_pair = klt.KLTPair(img_sequence, self._tracker, pair_name)

        intrinsic_mat = self._intrinsic_matrix
        img_1_pose_mat = self._pose_matrices[img_1_index]
        img_1_extrinsic_mat = self._extrinsic_matrices[img_1_index]
        img_2_pose_mat = self._pose_matrices[img_2_index]
        img_2_extrinsic_mat = self._extrinsic_matrices[img_2_index]

        # Both algorithms yield the same matrix when normalized
        if self._f_matrix_algorithm == calibrated.PINV_F_MAT_ALGORITHM:
            img_1_camera_center = img_1_pose_mat[0:3, -1, np.newaxis]
            img_2_camera_center = img_2_pose_mat[0:3, -1, np.newaxis]

            f_pair = calibrated.PinvFundamentalMatrixPair(
                klt_pair.image_1, klt_pair.image_2, klt_pair.name,
                img_1_camera_center, img_2_camera_center,
                intrinsic_mat @ img_1_extrinsic_mat, intrinsic_mat @ img_2_extrinsic_mat
            )
        elif self._f_matrix_algorithm == calibrated.STD_STEREO_F_MAT_ALGORITHM:
            intrinsic_mat_inv = self._intrinsic_matrix_inv
            f_pair = calibrated.StdStereoFundamentalMatrixPair(
                klt_pair.image_1, klt_pair.image_2, klt_pair.name,
                intrinsic_mat, intrinsic_mat,
                intrinsic_mat_inv, intrinsic_mat_inv,
                img_1_extrinsic_mat, img_2_extrinsic_mat,
                img_1_pose_mat, img_2_pose_mat
            )
        else:
            # Not reachable unless self._f_matrix_algorithm was tampered with
            assert False

        return pairs.CorrespondenceFundamentalMatrixPair(klt_pair, f_pair)

    def download(self: 'KITTIMonocularStereoPairsSequence') -> None:
        if not self._check_raw_exists():
            os.makedirs(self._raw_folder, exist_ok=True)
            url = self.url_color if self._color else self.url_gray
            tv_data.download_and_extract_archive(
                url, download_root=self._raw_folder, remove_finished=True
            )
            tv_data.download_and_extract_archive(
                self.url_pose, download_root=self._raw_folder, remove_finished=True
            )
            calibration_path = os.path.join(self._raw_folder, "calibration")
            os.makedirs(calibration_path)
            tv_data.download_and_extract_archive(
                self.url_calibration_map[self._sequence], download_root=calibration_path, remove_finished=True
            )
        if not self._check_processed_exists():
            os.makedirs(self._processed_sequence_folder, exist_ok=True)

            camera_dir = "image_2" if self._color else "image_0"
            old_image_path = os.path.join(self._raw_folder, "dataset", "sequences", self._sequence, camera_dir)
            new_image_path = os.path.join(self._processed_sequence_folder, "images")
            shutil.copytree(old_image_path, new_image_path)

            img_seq = sequence.GlobImageSequence(os.path.join(new_image_path, "*.png"))
            seq_overlap = self._tracker.find_sequence_overlap(img_seq, max_num_points=500)
            with open(os.path.join(self._processed_sequence_folder, "overlap.pickle"), 'wb') as overlap_file:
                pickle.dump(seq_overlap, overlap_file)

            # intrinsic_mats are the post rectification intrinsic matrices for each camera
            # relative extrinsic_mats is a stack of N 4x4 matrices which map points in the
            # rectified reference frame of camera 0 to the rectified reference frame
            # of camera n \in N
            calibration_file = os.path.join(self._raw_folder, self._file_calibration_map[self._sequence])
            calibration_data = self._read_calibration(calibration_file)
            intrinsic_mats, relative_extrinsic_mats = self._extract_camera_calibration(calibration_data)
            relative_pose_mats = np.linalg.inv(relative_extrinsic_mats)

            # cam_0_poses is a stack of T 4x4 matrices which map points in the rectified reference frame of
            # camera 0 at time t \in T to the rectified reference frame of camera 0 at time 0
            pose_file = os.path.join(self._raw_folder, "dataset", "poses", self._sequence + ".txt")
            cam_0_poses = np.loadtxt(pose_file, np.float64).reshape((-1, 3, 4))
            cam_0_poses = np.concatenate(
                (cam_0_poses, np.repeat(np.array([[[0, 0, 0, 1]]]), cam_0_poses.shape[0], axis=0)), axis=1)

            # select the left camera
            camera_idx = 2 if self._color else 0

            # derive the intrinsic and extrinsic matrices with respect to
            # camera 0 at time 0
            intrinsic_mat = intrinsic_mats[camera_idx]
            pose_mats = cam_0_poses @ relative_pose_mats[camera_idx]
            extrinsic_mats = np.linalg.inv(pose_mats)[:, 0:3, :]
            pose_mats = pose_mats[:, 0:3, :]

            calibration_data = (intrinsic_mat, pose_mats, extrinsic_mats)
            with open(os.path.join(self._processed_sequence_folder, "calibration_data.pickle"),
                      'wb') as calib_data_out_file_obj:
                pickle.dump(calibration_data, calib_data_out_file_obj)

    @staticmethod
    def _extract_camera_calibration(calibration_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Grab the post rectification intrinsic and extrinsic matrices
        # The extrinsic matrices returned here map points in
        # camera X reference frame to camera 0's reference frame.
        camera_ids = ["00", "01", "02", "03"]

        camera_mats = [
            calibration_data["P_rect_" + cam_id].reshape((3, 4))
            for cam_id in camera_ids
        ]

        intrinsic_mats = np.stack([
            camera_mat[0:3, 0:3]
            for camera_mat in camera_mats
        ])
        translation_x_vals = [
            camera_mat[0, 3] / camera_mat[0, 0]
            for camera_mat in camera_mats
        ]
        extrinsic_mats = np.stack([
            np.vstack(((1, 0, 0, x_val), np.eye(4)[1:, :]))
            for x_val in translation_x_vals
        ])
        return intrinsic_mats, extrinsic_mats

    @staticmethod
    def _read_calibration(calibration_file: str) -> Dict[str, np.ndarray]:
        data_dict = {}
        with open(calibration_file, 'r') as calib_file_obj:
            for line in calib_file_obj:
                key, value = line.split(':', 1)
                try:
                    data_dict[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data_dict

    def _check_raw_exists(self: 'KITTIMonocularStereoPairsSequence') -> bool:
        calibration_file = os.path.join(self._raw_folder, self._file_calibration_map[self._sequence])
        pose_file = os.path.join(self._raw_folder, "dataset", "poses", self._sequence + ".txt")
        camera_dir = "image_2" if self._color else "image_0"
        left_camera_images = os.path.join(self._raw_folder, "dataset", "sequences", self._sequence, camera_dir)

        return all([os.path.exists(file) for file in (calibration_file, pose_file, left_camera_images)])

    def _check_processed_exists(self: 'KITTIMonocularStereoPairsSequence') -> bool:
        calibration_file = os.path.join(self._processed_sequence_folder, "calibration_data.pickle")
        overlap_file = os.path.join(self._processed_sequence_folder, "overlap.pickle")
        left_camera_images = os.path.join(self._processed_sequence_folder, "images")
        return all([os.path.exists(file) for file in (calibration_file, overlap_file, left_camera_images)])


class KITTIMonocularStereoPairs(torch.utils.data.Dataset):
    all_sequences = [str(x).zfill(2) for x in range(11)]
    train_sequences = all_sequences[:5]
    test_sequences = all_sequences[5:]

    def __init__(self: 'KITTIMonocularStereoPairs',
                 root: str,
                 train: Optional[bool] = True,
                 download: Optional[bool] = True,
                 color: Optional[bool] = True,
                 minimum_KLT_overlap: Optional[float] = 0.3,
                 f_matrix_algorithm: Optional[int] = None) -> None:
        sequences = [KITTIMonocularStereoPairsSequence(
            root,
            seq,
            download,
            color,
            minimum_KLT_overlap,
            f_matrix_algorithm
        ) for seq in (self.train_sequences if train else self.test_sequences)]

        self._proxy = torch.utils.data.ConcatDataset(sequences)

    def __len__(self) -> int:
        return len(self._proxy)

    def __getitem__(self, index: int) -> pairs.CorrespondencePair:
        return self._proxy[index]
