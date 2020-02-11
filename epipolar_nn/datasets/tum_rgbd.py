import bisect
import os
import pickle
import shutil
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch.utils.data
import torchvision.datasets.utils as tv_data

from epipolar_nn.data import klt
from epipolar_nn.datasets import sequence


class TUMRGBDDataset:
    _FR3_DATASETS: List[str] = [
        "fr3/long_office_household",
        "fr3/nostructure_notexture_far",
        "fr3/nostructure_notexture_near_withloop",
        "fr3/nostructure_texture_far",
        "fr3/nostructure_texture_near_withloop",
        "fr3/structure_notexture_far",
        "fr3/structure_notexture_near",
        "fr3/structure_texture_far",
        "fr3/structure_texture_near",
        "fr3/sitting_static",
        "fr3/sitting_xyz",
        "fr3/sitting_halfsphere",
        "fr3/sitting_rpy",
        "fr3/walking_static",
        "fr3/walking_xyz",
        "fr3/walking_halfsphere",
        "fr3/walking_rpy",
        "fr3/cabinet",
        "fr3/large_cabinet",
        "fr3/teddy",
    ]

    _CAMERA_NAME_MAP: Dict[str, str] = {
        "fr1": "freiburg1",
        "fr2": "freiburg2",
        "fr3": "freiburg3",
    }

    _INTRINSICS_MAP: Dict[str, np.ndarray] = {
        "fr1": np.array([
            [517.306408, 0, 318.643040],
            [0, 516.469215, 255.313989],
            [0, 0, 1]
        ]),
        "fr2": np.array([
            [520.908620, 0, 325.141442],
            [0, 521.007327, 249.701764],
            [0, 0, 1]
        ]),
        "fr3": np.array([
            [537.960322, 0, 319.183641],
            [0, 539.597659, 247.053820],
            [0, 0, 1],
        ])
    }

    _RADIAL_DISTORTION_MAP: Dict[str, np.ndarray] = {
        "fr1": np.array([0.262383, -0.953104, -0.005358, 0.002628]),
        "fr2": np.array([0.231222, -0.784899, -0.003257, -0.000105]),
        "fr3": np.array([0, 0, 0, 0])
    }

    def __init__(self: 'TUMRGBDDataset', short_name: str):
        name_parts = short_name.split('/')
        camera_name = TUMRGBDDataset._CAMERA_NAME_MAP[name_parts[0]]
        self.short_name = short_name
        self.long_name = "rgbd_dataset_{0}_{1}".format(camera_name, name_parts[1])
        self.url = "https://vision.cs.tum.edu/rgbd/dataset/{0}/{1}.tgz".format(
            camera_name,
            self.long_name
        )
        self.intrinsic_matrix = TUMRGBDDataset._INTRINSICS_MAP[name_parts[0]]
        self.radial_distortion = TUMRGBDDataset._INTRINSICS_MAP[name_parts[0]]


class TUMRGBDStereoPairs(torch.utils.data.Dataset):

    @property
    def raw_folder(self: 'TUMRGBDStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'raw')

    @property
    def raw_extracted_folder(self: 'TUMRGBDStereoPairs') -> str:
        return os.path.join(self.raw_folder, self.dataset.long_name)

    @property
    def processed_folder(self: 'TUMRGBDStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'processed', self.dataset.long_name)

    def __init__(self: 'TUMRGBDStereoPairs', root: str,
                 dataset_name: str,
                 train: Optional[bool] = True,
                 download: Optional[bool] = False,
                 minimum_KLT_overlap: Optional[float] = 0.3) -> None:
        self.root_folder = os.path.abspath(root)
        self.dataset = TUMRGBDDataset(dataset_name)
        self.train = train

        self._tracker = klt.Tracker()

        if download:
            self.download()

        # load the pose data/ extrinsics map
        with open(os.path.join(self.processed_folder, "pose_data.pickle"), 'rb') as pose_data_file:
            pose_data = pickle.load(pose_data_file)
            image_names = pose_data[0]
            pose_matrices = pose_data[1]
            extrinsic_matrices = pose_data[2]
            self.extrinsics_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
                image_names[i]: (pose_matrices[i], extrinsic_matrices[i]) for i in range(len(image_names))
            }

        # load the images
        img_path = os.path.join(self.processed_folder, "images")
        self.image_sequence = sequence.GlobImageSequence(os.path.join(img_path, "*.png"))

        # If the image data doesn't match the pose data, quit.
        assert len(self.image_sequence) == len(image_names)

        # load the klt data
        with open(os.path.join(img_path, "overlap.pickle"), 'rb') as overlap_file:
            seq_overlap = pickle.load(overlap_file)

        self._overlapped_frames_cum_sum = []
        overlapped_frames = seq_overlap.find_frames_with_overlap(0, minimum_KLT_overlap).size
        self._overlapped_frames_cum_sum.append(overlapped_frames)
        for i in range(1, len(self.image_sequence) - 1):
            overlapped_frames = seq_overlap.find_frames_with_overlap(i, minimum_KLT_overlap).size
            self._overlapped_frames_cum_sum.append(overlapped_frames + self._overlapped_frames_cum_sum[-1])

    def __len__(self: 'TUMRGBDStereoPairs') -> int:
        return self._overlapped_frames_cum_sum[-1]

    def __getitem__(self, index: int):
        if index > len(self):
            raise IndexError()

        img_1_index = bisect.bisect_right(self._overlapped_frames_cum_sum, index)

        if img_1_index > 0:
            img_2_index = (img_1_index + 1) + (index - self._overlapped_frames_cum_sum[img_1_index - 1])
        else:
            img_2_index = 1 + index

        # TODO: Create type for deriving F matrix from extrinsics/ intrinsics/ etc.

        img_sequence = self.image_sequence[img_1_index:img_2_index + 1]  # Add 1 since the end of a slice is exclusive
        pair_name = "{0}: {1} {2}".format(self.dataset.short_name, img_1_index, img_2_index)

        img_1_extrinsics = self.extrinsics_map[self.image_sequence.file_name(img_1_index)]
        img_2_extrinsics = self.extrinsics_map[self.image_sequence.file_name(img_2_index)]
        intrinsics = self.dataset.intrinsic_matrix

        # return KLTPair(img_sequence, self._tracker, pair_name)

    def download(self):
        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            tv_data.download_url(
                self.dataset.url, root=self.raw_folder,
            )
            old_name = os.path.join(self.raw_folder, self.dataset.long_name + ".tgz", )
            new_name = os.path.join(self.raw_folder, self.dataset.long_name + ".tar.gz")
            os.rename(old_name, new_name)
            tv_data.extract_archive(new_name, self.raw_folder, remove_finished=True)
        if not self._check_processed_exists():
            os.makedirs(self.processed_folder, exist_ok=True)

            old_image_path = os.path.join(self.raw_extracted_folder, "rgb")
            new_image_path = os.path.join(self.processed_folder, "images")
            shutil.copytree(old_image_path, new_image_path)
            img_seq = sequence.GlobImageSequence(os.path.join(new_image_path, "*.png"))
            seq_overlap = self._tracker.find_sequence_overlap(img_seq, max_num_points=500)
            with open(os.path.join(new_image_path, "overlap.pickle"), 'wb') as overlap_file:
                pickle.dump(seq_overlap, overlap_file)

            pose_data = TUMRGBDStereoPairs._read_list_file(os.path.join(self.raw_extracted_folder, "groundtruth.txt"))
            image_timestamps = TUMRGBDStereoPairs._read_list_file(os.path.join(self.raw_extracted_folder, "rgb.txt"))
            images, poses, extrinsics, errors = TUMRGBDStereoPairs._extract_extrinsics(pose_data, image_timestamps)

            # Assert that there are no sampling issues in the middle of the dataset
            assert all([x[1] != TUMRGBDStereoPairs._ERR_IMG_TS_NO_SAMPLES] for x in errors)

            # If there are any images that were captured outside of the MoCap recording,
            # delete them.
            for error in errors:
                if error[1] == TUMRGBDStereoPairs._ERR_IMG_TS_OUT_OF_BOUNDS:
                    os.remove(os.path.join(self.processed_folder, "images", error[0]))

            pose_data = (images, poses, extrinsics)
            with open(os.path.join(self.processed_folder, "pose_data.pickle"), 'wb') as pose_data_file:
                pickle.dump(pose_data, pose_data_file)

    @staticmethod
    def _qvec2rotmat(quaternion: np.ndarray) -> np.ndarray:
        return np.array([
            [1 - 2 * quaternion[2] ** 2 - 2 * quaternion[3] ** 2,
             2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3],
             2 * quaternion[3] * quaternion[1] + 2 * quaternion[0] * quaternion[2]],
            [2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3],
             1 - 2 * quaternion[1] ** 2 - 2 * quaternion[3] ** 2,
             2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]],
            [2 * quaternion[3] * quaternion[1] - 2 * quaternion[0] * quaternion[2],
             2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1],
             1 - 2 * quaternion[1] ** 2 - 2 * quaternion[2] ** 2]])

    _ERR_IMG_TS_OUT_OF_BOUNDS = 0
    _ERR_IMG_TS_NO_SAMPLES = 1

    @staticmethod
    def _extract_extrinsics(pose_data: Dict[float, str],
                            image_timestamps: Dict[float, str],
                            max_time_difference: Optional[float] = 0.02) -> Tuple[
        List[str], List[np.ndarray], List[np.ndarray], List[Tuple[str, int]]]:
        pose_timestamps = sorted(pose_data.keys())
        image_names: List[str] = []
        poses: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        errors: List[Tuple[str, int]] = []
        for image_ts in sorted(image_timestamps.keys()):
            image_name = image_timestamps[image_ts]
            image_name = image_name.split("/")[-1]

            # we need to find two samples around the image's timestamp
            # pose_ts_low is <= image_ts and pose_ts_high_idx > image_ts
            pose_ts_low_idx = bisect.bisect_right(pose_timestamps, image_ts) - 1
            pose_ts_high_idx = pose_ts_low_idx + 1

            # bounds check in case the bisect ends on the first or last value
            if pose_ts_low_idx < 0 or pose_ts_high_idx >= len(pose_timestamps):
                errors.append((image_name, TUMRGBDStereoPairs._ERR_IMG_TS_OUT_OF_BOUNDS))
                continue

            pose_ts_low = pose_timestamps[pose_ts_low_idx]
            pose_ts_high = pose_timestamps[pose_ts_high_idx]

            # Ensure the samples are near the image time stamp
            if image_ts - pose_ts_low > max_time_difference:
                errors.append((image_name, TUMRGBDStereoPairs._ERR_IMG_TS_NO_SAMPLES))
                continue

            if pose_ts_high - image_ts > max_time_difference:
                errors.append((image_name, TUMRGBDStereoPairs._ERR_IMG_TS_NO_SAMPLES))
                continue

            # Run a linear interpolation between the two poses
            t = (image_ts - pose_ts_low) / (pose_ts_high - pose_ts_low)
            pose_low = np.fromstring(pose_data[pose_ts_low], sep=" ")
            pose_high = np.fromstring(pose_data[pose_ts_high], sep=" ")
            pose_interp = (1 - t) * pose_low + t * pose_high

            # Normalize the quaternion pose back to the unit sphere
            pose_interp[3:7] = pose_interp[3:7] / np.linalg.norm(pose_interp[3:7], ord=2)

            # Convert the pose to an extrinsic matrix
            # Convert the quaternion pose to a rotation matrix
            pose_rotation_matrix = TUMRGBDStereoPairs._qvec2rotmat(pose_interp[3:7])
            pose_t_vec = np.reshape(pose_interp[0:3], (3, 1))
            pose_matrix = np.block([
                [pose_rotation_matrix, pose_t_vec]
            ])
            # The matrix [pose_rotation_matrix -- 0T | pose_t_vec -- 1] can be viewed as an affine transform
            # from camera coordinates to world coordinates. We need to invert this affine transform
            # in order to find the extrinsic matrix which transforms world coordinates to camera coordinates.
            extrinsic_matrix = np.block([
                [pose_rotation_matrix.T, -1 * pose_rotation_matrix.T @ pose_t_vec]
            ])
            image_names.append(image_name)
            poses.append(pose_matrix)
            extrinsics.append(extrinsic_matrix)
        return image_names, poses, extrinsics, errors

    @staticmethod
    def _read_list_file(path: str) -> Dict[float, str]:
        file = open(path)
        data = file.read()
        file.close()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        kvp_list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                    len(line) > 0 and line[0] != "#"]
        kvp_list = [(float(line[0]), " ".join(line[1:])) for line in kvp_list if len(line) > 1]
        return dict(kvp_list)

    def _check_raw_exists(self: 'TUMRGBDStereoPairs') -> bool:
        return os.path.exists(self.raw_extracted_folder)

    def _check_processed_exists(self: 'TUMRGBDStereoPairs') -> bool:
        return all([
            os.path.exists(os.path.join(self.processed_folder, resource)) for resource in [
                "images",
                os.path.join("images", "overlap.pickle"),
                "pose_data.pickle",
            ]
        ])
