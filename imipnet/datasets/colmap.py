import os
import sqlite3
import struct
from typing import Optional, Tuple, BinaryIO

import cv2
import numpy as np
import torch.utils.data

from . import colmap_read
from ..data import pairs, aflow, calibrated


class COLMAPStereoPairs(torch.utils.data.Dataset):

    @property
    def _project_folder(self) -> str:
        return os.path.join(self._root_folder, self.__class__.__name__, self._colmap_project)

    @property
    def _images_folder(self) -> str:
        return os.path.join(self._project_folder, "dense", "images")

    @property
    def _sparse_folder(self) -> str:
        return os.path.join(self._project_folder, "dense", "sparse")

    @property
    def _depth_maps_folder(self) -> str:
        return os.path.join(self._project_folder, "dense", "stereo", "depth_maps")

    @property
    def _database_path(self) -> str:
        return os.path.join(self._project_folder, self._db_name)

    @property
    def _pair_index_path(self) -> str:
        pair_file = f"pair_ids_min_{self._minimum_SIFT_matches}_matches"
        if self._max_image_bytes is not None:
            color_tag = "color" if self._color else "gray"
            pair_file += f"_max_{self._max_image_bytes}_{color_tag}_image_bytes"
        pair_file += ".bin"

        return os.path.join(self._project_folder, pair_file)

    def __init__(self, data_root: str, colmap_project_name: str, color: Optional[bool] = True,
                 minimum_matches: Optional[int] = 1024, db_name: Optional[str] = "colmap.db",
                 max_image_bytes=None):
        self._root_folder = os.path.abspath(data_root)
        self._colmap_project = colmap_project_name
        self._db_name = db_name
        self._color = color
        self._max_image_bytes = max_image_bytes

        self._cameras = colmap_read.read_cameras_binary(os.path.join(self._sparse_folder, "cameras.bin"))
        self._images = colmap_read.read_images_binary(os.path.join(self._sparse_folder, "images.bin"))
        self._minimum_SIFT_matches = minimum_matches
        if not self._check_index_exists():
            self._generate_flow_pair_index(minimum_matches)

        with open(self._pair_index_path, 'rb') as pairs_index_file:
            self._pairs_index = COLMAPStereoPairs._load_pairs_index(pairs_index_file)

    def __len__(self) -> int:
        return len(self._pairs_index)

    def __getitem__(self, i: int) -> pairs.CorrespondenceFundamentalMatrixPair:
        if i >= len(self):
            raise IndexError()

        pair_id = self._pairs_index[i]
        image_1_id, image_2_id = self._pair_id_to_image_ids(pair_id)

        # load image 1 calibration
        image_1_data = self._images[image_1_id]
        camera_1_data = self._cameras[image_1_data.camera_id]
        image_1_intrinsic, image_1_extrinsic, image_1_shape = COLMAPStereoPairs._derive_camera_data(
            camera_1_data, image_1_data
        )
        image_1_camera_matrix = image_1_intrinsic @ image_1_extrinsic[:3]

        # load image 1 depth map
        image_1_depth_map_path = os.path.join(self._depth_maps_folder, image_1_data.name + ".geometric.bin")
        image_1_depth_map = colmap_read.read_depth_map(image_1_depth_map_path)
        image_1_depth_map[image_1_depth_map == 0] = float('nan')  # convert missing marker to NaN

        # load image 2 calibration
        image_2_data = self._images[image_2_id]
        camera_2_data = self._cameras[image_2_data.camera_id]
        image_2_intrinsic, image_2_extrinsic, image_2_shape = COLMAPStereoPairs._derive_camera_data(
            camera_2_data, image_2_data,
        )
        image_2_camera_matrix = image_2_intrinsic @ image_2_extrinsic[:3]

        # load image 2 depth map
        image_2_depth_map_path = os.path.join(self._depth_maps_folder, image_2_data.name + ".geometric.bin")
        image_2_depth_map = colmap_read.read_depth_map(image_2_depth_map_path)
        image_2_depth_map[image_2_depth_map == 0] = float('nan')

        # load the actual images
        image_1 = cv2.imread(
            os.path.join(self._images_folder, image_1_data.name),
            cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
        )

        image_2 = cv2.imread(
            os.path.join(self._images_folder, image_2_data.name),
            cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
        )

        if self._color:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

        # give the current pair a name
        pair_name = "COLMAP {0}: {1} {2}".format(self._colmap_project, image_1_data.name, image_2_data.name)

        # construct an absolute flow pair from camera matrices and depth maps
        absolute_flow_pair = aflow.CalibratedDepthPair(
            image_1, image_2, pair_name,
            image_1_camera_matrix, image_2_camera_matrix,
            image_1_depth_map, image_2_depth_map,
        )
        f_pair = calibrated.StdStereoFundamentalMatrixPair(
            image_1, image_2, pair_name,
            image_1_intrinsic, image_2_intrinsic,
            np.linalg.inv(image_1_intrinsic), np.linalg.inv(image_2_intrinsic),
            image_1_extrinsic[:3], image_2_extrinsic[:3],
            np.linalg.inv(image_1_extrinsic)[:3], np.linalg.inv(image_2_extrinsic)[:3]
        )

        return pairs.CorrespondenceFundamentalMatrixPair(absolute_flow_pair, f_pair)

    def _generate_flow_pair_index(self, minimum_matches: int):
        print("Generating pairs index for COLMAP dataset: {0}".format(self._database_path))

        with open(self._pair_index_path, 'wb') as index_file:
            index_file.write(struct.pack('Q', 0))  # Make space for writing total count of pair ids

            connection = sqlite3.connect(self._database_path)
            cursor = connection.cursor()
            num_pairs = 0  # 154498 pairs for 0000
            cursor.execute(
                "SELECT pair_id FROM two_view_geometries WHERE rows>=? AND config >=2;", (minimum_matches,)
            )
            for row in cursor:
                pair_id = row[0]
                # Filter out the pair ids which we relate images not in the sparse data
                image_1_id, image_2_id = COLMAPStereoPairs._pair_id_to_image_ids(pair_id)
                try:
                    image_1 = self._images[image_1_id]
                    image_2 = self._images[image_2_id]

                    image_1_bytes = cv2.imread(
                        os.path.join(self._images_folder, image_1.name),
                        cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
                    ).size

                    if self._max_image_bytes is not None and image_1_bytes > self._max_image_bytes:
                        continue

                    image_2_bytes = cv2.imread(
                        os.path.join(self._images_folder, image_2.name),
                        cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
                    ).size

                    if self._max_image_bytes is not None and image_2_bytes > self._max_image_bytes:
                        continue

                except KeyError:
                    continue
                # Save pair id as uint32 to index file
                index_file.write(struct.pack('Q', pair_id))
                num_pairs += 1
            index_file.seek(0)
            index_file.write(struct.pack('Q', num_pairs))
            connection.close()

        print("Created pair index for COLMAP dataset with {0} pairs".format(num_pairs))

    @staticmethod
    def _derive_camera_data(camera: colmap_read.Camera, image: colmap_read.Image) -> \
            Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        # returns intrinsic matrix, extrinsic matrix and image shape
        intrinsic_matrix = np.eye(3)
        intrinsic_matrix[0, 0] = camera.params[0]
        intrinsic_matrix[1, 1] = camera.params[1]
        intrinsic_matrix[0, 2] = camera.params[2]
        intrinsic_matrix[1, 2] = camera.params[3]

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = image.qvec2rotmat()
        extrinsic_matrix[:3, 3] = image.tvec
        return intrinsic_matrix, extrinsic_matrix, (camera.height, camera.width)

    @staticmethod
    def _load_pairs_index(pairs_index_file: BinaryIO):
        pairs_index_file.seek(0)
        num_pairs = struct.unpack('Q', pairs_index_file.read(struct.calcsize('Q')))[0]
        return struct.unpack('Q' * num_pairs, pairs_index_file.read())

    @staticmethod
    def _pair_id_to_image_ids(pair_id):
        # as per colmap docs
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) / 2147483647
        return int(image_id1), int(image_id2)

    def _check_index_exists(self) -> bool:
        return os.path.exists(self._pair_index_path) and os.path.getsize(self._pair_index_path) > 0
