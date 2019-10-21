import numpy as np
import glob
import cv2


class ImageSequence:
    def __getitem__(self, index: int) -> np.ndarray:
        pass

    def __len__(self) -> int:
        pass


class DirectoryImageSequence(ImageSequence):
    def __init__(self: 'DirectoryImageSequence', glob_path: str, recursive=False, convert_to_grayscale=False) -> None:
        self._file_paths = sorted(glob.glob(glob_path, recursive=recursive))
        self.convert_to_grayscale = convert_to_grayscale

    def __getitem__(self, index: int) -> np.ndarray:
        if self.convert_to_grayscale:
            return cv2.imread(self._file_paths[index], cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(self._file_paths[index], cv2.IMREAD_COLOR)

    def __len__(self) -> int:
        return len(self._file_paths)