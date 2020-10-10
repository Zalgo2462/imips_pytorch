import os
import sqlite3
import struct
from typing import Optional, Tuple, BinaryIO, Union

import cv2
import numpy as np
import torch.utils.data

from . import colmap_read
from ..data import pairs, aflow, calibrated


class COLMAPStereoPairs(torch.utils.data.Dataset):
    class PairData:
        def __init__(self, image_1_name: Union[str, np.array], image_2_name: Union[str, np.array],
                     image_1_intrinsic: np.ndarray, image_1_extrinsic: np.ndarray,
                     image_1_intrinsic_inv: np.ndarray, image_1_pose: np.ndarray,
                     image_2_intrinsic: np.ndarray, image_2_extrinsic: np.ndarray,
                     image_2_intrinsic_inv: np.ndarray, image_2_pose: np.ndarray,
                     image_1_aflow: np.ndarray, image_2_aflow: np.ndarray
                     ):
            self.image_1_name = str(image_1_name)
            self.image_2_name = str(image_2_name)
            self.image_1_intrinsic = image_1_intrinsic
            self.image_1_extrinsic = image_1_extrinsic
            self.image_1_intrinsic_inv = image_1_intrinsic_inv
            self.image_1_pose = image_1_pose
            self.image_2_intrinsic = image_2_intrinsic
            self.image_2_extrinsic = image_2_extrinsic
            self.image_2_intrinsic_inv = image_2_intrinsic_inv
            self.image_2_pose = image_2_pose
            self.image_1_aflow = image_1_aflow
            self.image_2_aflow = image_2_aflow

        @staticmethod
        def from_COLMAP_data(image_1: colmap_read.Image, image_2: colmap_read.Image,
                             camera_1: colmap_read.Camera, camera_2: colmap_read.Camera,
                             image_1_depth_map: np.ndarray,
                             image_2_depth_map: np.ndarray) -> 'COLMAPStereoPairs.PairData':
            image_1_intrinsic, image_1_extrinsic, image_1_shape = COLMAPStereoPairs.PairData._derive_camera_data(
                camera_1, image_1
            )
            image_2_intrinsic, image_2_extrinsic, image_2_shape = COLMAPStereoPairs.PairData._derive_camera_data(
                camera_2, image_2
            )

            # create projection and inverse projection ala
            # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/mvs/fusion.cc#L373

            image_1_camera_matrix = image_1_intrinsic @ image_1_extrinsic[:3]
            image_1_camera_matrix_inv = np.linalg.inv(np.block([[image_1_camera_matrix], [np.array([0, 0, 0, 1])]]))

            image_2_camera_matrix = image_2_intrinsic @ image_2_extrinsic[:3]
            image_2_camera_matrix_inv = np.linalg.inv(np.block([[image_2_camera_matrix], [np.array([0, 0, 0, 1])]]))

            image_1_aflow = COLMAPStereoPairs.PairData._calculate_absolute_flow(
                image_1_camera_matrix_inv, image_2_camera_matrix, image_1_depth_map, image_1_shape, image_2_shape
            )

            image_2_aflow = COLMAPStereoPairs.PairData._calculate_absolute_flow(
                image_2_camera_matrix_inv, image_1_camera_matrix, image_2_depth_map, image_2_shape, image_1_shape
            )

            image_1_aflow, image_2_aflow = COLMAPStereoPairs.PairData._refine_pairwise_absolute_flow(
                image_1_aflow, image_2_aflow
            )

            return COLMAPStereoPairs.PairData(image_1_name=image_1.name, image_2_name=image_2.name,
                                              image_1_intrinsic=image_1_intrinsic, image_1_extrinsic=image_1_extrinsic,
                                              image_1_intrinsic_inv=np.linalg.inv(image_1_intrinsic),
                                              image_1_pose=np.linalg.inv(image_1_extrinsic),
                                              image_2_intrinsic=image_2_intrinsic, image_2_extrinsic=image_2_extrinsic,
                                              image_2_intrinsic_inv=np.linalg.inv(image_2_intrinsic),
                                              image_2_pose=np.linalg.inv(image_2_extrinsic),
                                              image_1_aflow=image_1_aflow, image_2_aflow=image_2_aflow)

        @staticmethod
        def _calculate_absolute_flow(image_1_camera_matrix_inv: np.ndarray, image_2_camera_matrix: np.ndarray,
                                     image_1_depth_map: np.ndarray, image_1_shape: Tuple[int, int],
                                     image_2_shape: Tuple[int, int]) -> np.ndarray:
            # camera_inv @ [col * depth, row * depth, depth, 1].T
            x, y = np.meshgrid(np.arange(image_1_shape[1], dtype=np.float32),
                               np.arange(image_1_shape[0], dtype=np.float32))
            ones = np.ones_like(x)
            xy = np.stack((x, y, ones, ones), axis=0)
            xy[:3, :, :] = xy[:3, :, :] * np.expand_dims(image_1_depth_map, axis=0)

            abs_flow = (image_2_camera_matrix @ (image_1_camera_matrix_inv @ xy.reshape(4, -1)))
            abs_flow = abs_flow[0:2] / abs_flow[2:3]

            with np.errstate(invalid='ignore'):  # silence warnings stemming from nan comparisons
                bad_flow = ((abs_flow[0, :] < 0) | (abs_flow[1, :] < 0) |
                            (abs_flow[0, :] >= image_2_shape[1]) | (abs_flow[1, :] >= image_2_shape[0]))

            bad_flow = np.stack((bad_flow, bad_flow), axis=0)
            abs_flow[bad_flow] = float('nan')

            abs_flow = abs_flow.reshape(2, image_1_shape[0], image_1_shape[1])
            return abs_flow.astype(np.float32)

        @staticmethod
        def _refine_pairwise_absolute_flow(image_1_abs_flow: np.ndarray, image_2_abs_flow: np.ndarray):
            image_1_flow_shape = image_1_abs_flow.shape
            image_2_flow_shape = image_2_abs_flow.shape

            # shift to a linear perspective to make the indexing easier
            image_1_flow_linear = image_1_abs_flow.reshape(2, -1)
            image_2_flow_linear = image_2_abs_flow.reshape(2, -1)

            # remove the nans, but keep the index so we can remake the flow
            image_1_non_nan_index = np.arange(image_1_flow_linear.shape[1])[~np.isnan(image_1_flow_linear[0])]
            image_1_flow_filtered = image_1_flow_linear[:, image_1_non_nan_index]

            image_2_non_nan_index = np.arange(image_2_flow_linear.shape[1])[~np.isnan(image_2_flow_linear[0])]
            image_2_flow_filtered = image_2_flow_linear[:, image_2_non_nan_index]

            # round accounting for border pixels
            image_1_flow_filtered = np.round(image_1_flow_filtered).astype(np.int)
            image_1_flow_filtered[0, image_1_flow_filtered[0] == image_2_flow_shape[2]] -= 1
            image_1_flow_filtered[1, image_1_flow_filtered[1] == image_2_flow_shape[1]] -= 1

            image_2_flow_filtered = np.round(image_2_flow_filtered).astype(np.int)
            image_2_flow_filtered[0, image_2_flow_filtered[0] == image_1_flow_shape[2]] -= 1
            image_2_flow_filtered[1, image_2_flow_filtered[1] == image_1_flow_shape[1]] -= 1

            # check if indexing the other flow field with the first field results in a nan flow
            image_1_bad_reverse_flow = np.isnan(
                image_2_abs_flow[0, image_1_flow_filtered[1], image_1_flow_filtered[0]]
            )
            image_1_flow_linear[:, image_1_non_nan_index[image_1_bad_reverse_flow]] = float('nan')

            image_2_bad_reverse_flow = np.isnan(
                image_1_abs_flow[0, image_2_flow_filtered[1], image_2_flow_filtered[0]]
            )
            image_2_flow_linear[:, image_2_non_nan_index[image_2_bad_reverse_flow]] = float('nan')

            image_1_abs_flow = image_1_flow_linear.reshape(image_1_flow_shape)
            image_2_abs_flow = image_2_flow_linear.reshape(image_2_flow_shape)

            return image_1_abs_flow, image_2_abs_flow

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

        def save(self, file: BinaryIO):
            np.savez_compressed(file, **self.__dict__)

        @staticmethod
        def load(file: BinaryIO):
            return COLMAPStereoPairs.PairData(**dict(np.load(file).items()))

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
        return os.path.join(self._project_folder, f"pair_ids_min_{self._minimum_SIFT_matches}_matches.bin")

    def __init__(self, root: str, colmap_project_name: str, color: Optional[bool] = True,
                 minimum_matches: Optional[int] = 1024, db_name: Optional[str] = "colmap.db"):
        self._root_folder = os.path.abspath(root)
        self._colmap_project = colmap_project_name
        self._db_name = db_name
        self._color = color

        self._cameras = colmap_read.read_cameras_binary(os.path.join(self._sparse_folder, "cameras.bin"))
        self._images = colmap_read.read_images_binary(os.path.join(self._sparse_folder, "images.bin"))
        self._minimum_SIFT_matches = minimum_matches
        if not self._check_index_exists():
            self._generate_flow_pair_index(minimum_matches)

        self._pairs_index_file = open(self._pair_index_path, 'rb')
        self._num_pairs = COLMAPStereoPairs._read_num_pairs_from_index(self._pairs_index_file)

    def __del__(self):
        if (hasattr(self, "_pairs_index_file") and self._pairs_index_file is not None and
                not self._pairs_index_file.closed):
            self._pairs_index_file.close()

    def __len__(self) -> int:
        return self._num_pairs

    def __getitem__(self, i: int) -> pairs.CorrespondenceFundamentalMatrixPair:
        if i > len(self):
            raise IndexError()

        pair_id = COLMAPStereoPairs._get_pair_id(self._pairs_index_file, i)
        image_1_id, image_2_id = self._pair_id_to_image_ids(pair_id)

        image_1_data = self._images[image_1_id]
        camera_1_data = self._cameras[image_1_data.camera_id]
        image_1_depth_map_path = os.path.join(self._depth_maps_folder, image_1_data.name + ".geometric.bin")
        image_1_depth_map = colmap_read.read_depth_map(image_1_depth_map_path)
        image_1_depth_map[image_1_depth_map == 0] = float('nan')

        image_2_data = self._images[image_2_id]
        camera_2_data = self._cameras[image_2_data.camera_id]
        image_2_depth_map_path = os.path.join(self._depth_maps_folder, image_2_data.name + ".geometric.bin")
        image_2_depth_map = colmap_read.read_depth_map(image_2_depth_map_path)
        image_2_depth_map[image_2_depth_map == 0] = float('nan')

        # the ids match the data in the database
        # the image and camera data is properly loaded

        pair_data = COLMAPStereoPairs.PairData.from_COLMAP_data(
            image_1_data, image_2_data,
            camera_1_data, camera_2_data,
            image_1_depth_map, image_2_depth_map
        )

        image_1 = cv2.imread(
            os.path.join(self._images_folder, pair_data.image_1_name),
            cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
        )

        image_2 = cv2.imread(
            os.path.join(self._images_folder, pair_data.image_2_name),
            cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
        )

        if self._color:
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

        pair_name = "COLMAP {0}: {1} {2}".format(self._colmap_project, pair_data.image_1_name, pair_data.image_2_name)

        absolute_flow_pair = aflow.AbsoluteFlowPair(image_1, image_2, pair_name,
                                                    pair_data.image_1_aflow, pair_data.image_2_aflow)

        f_pair = calibrated.StdStereoFundamentalMatrixPair(
            image_1, image_2, pair_name,
            pair_data.image_1_intrinsic, pair_data.image_2_intrinsic,
            pair_data.image_1_intrinsic_inv, pair_data.image_2_intrinsic_inv,
            pair_data.image_1_extrinsic[:3], pair_data.image_2_extrinsic[:3],
            pair_data.image_1_pose[:3], pair_data.image_2_pose[:3]
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
                    _ = self._images[image_1_id]
                    _ = self._images[image_2_id]
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
    def _read_num_pairs_from_index(pairs_index_file: BinaryIO) -> int:
        pairs_index_file.seek(0)
        return struct.unpack('Q', pairs_index_file.read(struct.calcsize('Q')))[0]

    @staticmethod
    def _get_pair_id(pairs_index_file: BinaryIO, i: int) -> int:
        pairs_index_file.seek(struct.calcsize('Q') * (i + 1))
        return struct.unpack('Q', pairs_index_file.read(struct.calcsize('Q')))[0]

    @staticmethod
    def _pair_id_to_image_ids(pair_id):
        # as per colmap docs
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) / 2147483647
        return int(image_id1), int(image_id2)

    def _check_index_exists(self) -> bool:
        return os.path.exists(self._pair_index_path) and os.path.getsize(self._pair_index_path) > 0
