import os
import tarfile
from typing import Optional

import cv2
import numpy as np
import torch.utils.data
import torchvision.datasets.utils as tv_data

from imipnet.data.aflow import CalibratedDepthPair
from imipnet.data.calibrated import StdStereoFundamentalMatrixPair
from imipnet.data.pairs import CorrespondenceFundamentalMatrixPair
from .blender_pairs_read import read_pairs_index


class BlenderStereoPairs(torch.utils.data.Dataset):
    dataset_name_url_map = {
        "livingroom_1": "https://github.com/Zalgo2462/BlenderStereoPairs/releases/download/v0.0.1/livingroom_1.tgz",
        "livingroom_2": "https://github.com/Zalgo2462/BlenderStereoPairs/releases/download/v0.0.1/livingroom_2.tgz",
        "livingroom_3": "https://github.com/Zalgo2462/BlenderStereoPairs/releases/download/v0.0.1/livingroom_3.tgz"
    }

    @property
    def _project_folder(self) -> str:
        return os.path.join(self._root_folder, self.__class__.__name__, self._dataset_name)

    @property
    def _pairs_index_path(self) -> str:
        return os.path.join(self._project_folder, "pairs.csv")

    def __init__(self, data_root: str, dataset_name: str, download: Optional[bool] = True,
                 color: Optional[bool] = True):
        self._root_folder = os.path.abspath(data_root)
        self._dataset_name = dataset_name
        self._color = color

        if download:
            self.download()

        with open(self._pairs_index_path, 'r') as pair_index_fd:
            self._pairs_index = read_pairs_index(pair_index_fd)

    def __len__(self) -> int:
        return len(self._pairs_index)

    def __getitem__(self, i: int) -> CorrespondenceFundamentalMatrixPair:
        if i >= len(self):
            raise IndexError()

        blender_dat_1, blender_dat_2 = self._pairs_index[i].load(self._project_folder)

        pair_name = "Blender {0}: {1} {2}".format(self._dataset_name, blender_dat_1.name, blender_dat_2.name)

        # handle color format conversions
        image_1 = blender_dat_1.image
        image_2 = blender_dat_2.image

        if self._color:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
        else:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        # convert no-data markers in depth map
        image_1_depth = blender_dat_1.depth_map
        image_1_depth[image_1_depth == 10000000000] = float('nan')

        image_2_depth = blender_dat_2.depth_map
        image_2_depth[image_2_depth == 10000000000] = float('nan')

        """
        # this seems to make the depth maps worse...
        # https://blender.stackexchange.com/questions/130970/cycles-generates-distorted-depth
        # https://devtalk.blender.org/t/cycles-generates-distorted-depth/5409
        # convert straight line depths from camera center to points into rectilinear depth
        image_1_depth = self._rectify_blender_depth_map(
            image_1_depth, blender_dat_1.intrinsic_matrix[0, 0], blender_dat_1.intrinsic_matrix[1, 1]
        )
        image_2_depth = self._rectify_blender_depth_map(
            image_2_depth, blender_dat_2.intrinsic_matrix[0, 0], blender_dat_2.intrinsic_matrix[1, 1]
        )
        """

        # construct an absolute flow pair from camera matrices and depth maps
        image_1_camera_matrix = blender_dat_1.intrinsic_matrix @ blender_dat_1.extrinsic_matrix
        image_2_camera_matrix = blender_dat_2.intrinsic_matrix @ blender_dat_2.extrinsic_matrix
        absolute_flow_pair = CalibratedDepthPair(
            image_1, image_2, pair_name,
            image_1_camera_matrix, image_2_camera_matrix,
            image_1_depth, image_2_depth
        )

        image_1_intrinsic_matrix_inv = np.linalg.inv(blender_dat_1.intrinsic_matrix)
        image_1_pose_matrix = np.linalg.inv(np.block([[blender_dat_1.extrinsic_matrix], [np.array([0, 0, 0, 1])]]))[:3]

        image_2_intrinsic_matrix_inv = np.linalg.inv(blender_dat_2.intrinsic_matrix)
        image_2_pose_matrix = np.linalg.inv(np.block([[blender_dat_2.extrinsic_matrix], [np.array([0, 0, 0, 1])]]))[:3]
        f_pair = StdStereoFundamentalMatrixPair(
            image_1, image_2, pair_name,
            blender_dat_1.intrinsic_matrix, blender_dat_2.intrinsic_matrix,
            image_1_intrinsic_matrix_inv, image_2_intrinsic_matrix_inv,
            blender_dat_1.extrinsic_matrix, blender_dat_2.extrinsic_matrix,
            image_1_pose_matrix, image_2_pose_matrix
        )

        return CorrespondenceFundamentalMatrixPair(absolute_flow_pair, f_pair)

    def download(self):
        if os.path.exists(self._project_folder):
            return

        url = self.dataset_name_url_map[self._dataset_name]
        target_dir = os.path.join(self._root_folder, self.__class__.__name__)
        file_name = os.path.basename(url)
        archive_file = os.path.join(target_dir, file_name)
        tv_data.download_url(url, target_dir, file_name)
        with tarfile.open(archive_file, 'r:gz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=target_dir)
        os.remove(archive_file)

        return os.path.exists(self._project_folder)

    """
    @staticmethod
    def _rectify_blender_depth_map(depth: np.ndarray, f_u: float, f_v: float) -> np.ndarray:
        x_axis = (np.arange(0, depth.shape[1]) + 0.5) / depth.shape[1] - 0.5  # need to test flipping subtraction
        y_axis = (np.arange(0, depth.shape[0]) + 0.5) / depth.shape[0] - 0.5

        x_axis *= depth.shape[1] / f_u  # Equiv. using blender details: x_axis *= sensor_width / lens_size
        y_axis *= depth.shape[0] / f_v  # Equiv. using blender details: y_axis *= sensor_height / lens_size

        x_chan, y_chan = np.meshgrid(x_axis, y_axis)
        z_chan = np.ones_like(x_chan)
        xyz = np.stack((x_chan, y_chan, z_chan), axis=0)
        xyz /= np.linalg.norm(xyz, ord=2, axis=0)
        depth = xyz[2] * depth
        return depth
    """
