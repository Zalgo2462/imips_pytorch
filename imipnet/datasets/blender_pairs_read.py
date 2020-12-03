import collections
import csv
import os
from typing import IO, List, Tuple

import OpenEXR as exr
import numpy as np

BlenderImage = collections.namedtuple("BlenderImage", [
    "image",
    "depth_map",
    "intrinsic_matrix",
    "extrinsic_matrix",
    "name"
])

EXRImage = collections.namedtuple("EXRImage", ["bgr", "depth"])


def read_exr_image(exr_path: str) -> EXRImage:
    """
    reads an exr image exported by Blender as an sRGB bgr image as returned by cv2.imread and a depth map
    :param exr_path: path to exr file
    :return: EXR image with bgr image and depth map
    """
    fd = exr.InputFile(exr_path)
    disp_window = fd.header()['displayWindow']
    shape = (disp_window.max.y - disp_window.min.y + 1, disp_window.max.x - disp_window.min.x + 1)
    depth = np.fromstring(fd.channel("Depth.V"), np.float32).reshape(shape)
    bgr = np.stack((
        np.fromstring(fd.channel("Image.B"), np.float32).reshape(shape),
        np.fromstring(fd.channel("Image.G"), np.float32).reshape(shape),
        np.fromstring(fd.channel("Image.R"), np.float32).reshape(shape),
    ), axis=-1)
    fd.close()

    # convert HDR linear data into sRGB data with gamma correction
    # https://en.wikipedia.org/wiki/SRGB
    low_domain = bgr <= 0.0031308
    high_domain = ~low_domain
    bgr[low_domain] *= 0.0031308
    bgr[high_domain] = 1.055 * pow(bgr[high_domain], 1.0 / 2.4) - 0.055
    bgr[bgr >= 1] = 1
    bgr = np.round(bgr * 255).astype(np.uint8)

    return EXRImage(bgr, depth)


class PairIndex:
    def __init__(self, camera_1: str, frame_1: int, camera_2: str, frame_2: int):
        self.camera_1 = camera_1
        self.frame_1 = frame_1
        self.camera_2 = camera_2
        self.frame_2 = frame_2

    def load(self, data_root: str) -> Tuple[BlenderImage, BlenderImage]:
        name_1 = "%s:%04d" % (self.camera_1, self.frame_1)

        name_2 = "%s:%04d" % (self.camera_2, self.frame_2)

        exr_path_1 = os.path.join(data_root, "EXR", self.camera_1, "%04d.exr" % self.frame_1)
        img_1, depth_1 = read_exr_image(exr_path_1)

        exr_path_2 = os.path.join(data_root, "EXR", self.camera_2, "%04d.exr" % self.frame_2)
        img_2, depth_2 = read_exr_image(exr_path_2)

        intrinsics_path_1 = os.path.join(data_root, "K", self.camera_1 + ".txt")
        K_1 = np.loadtxt(intrinsics_path_1)

        intrinsics_path_2 = os.path.join(data_root, "K", self.camera_2 + ".txt")
        K_2 = np.loadtxt(intrinsics_path_2)

        extrinsics_path_1 = os.path.join(data_root, "RT", self.camera_1, "%04d.txt" % self.frame_1)
        RT_1 = np.loadtxt(extrinsics_path_1)

        extrinsics_path_2 = os.path.join(data_root, "RT", self.camera_2, "%04d.txt" % self.frame_2)
        RT_2 = np.loadtxt(extrinsics_path_2)

        return (
            BlenderImage(img_1, depth_1, K_1, RT_1, name_1),
            BlenderImage(img_2, depth_2, K_2, RT_2, name_2),
        )


def read_pairs_index(fd: IO) -> List[PairIndex]:
    csv_reader = csv.reader(fd)
    pairs = []
    for row in csv_reader:
        pairs.append(PairIndex(row[0], int(row[1]), row[2], int(row[3])))
    return pairs
