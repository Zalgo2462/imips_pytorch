import os
import pickle
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch.utils.data
import torchvision.datasets.utils as tv_data

from imipnet.data import planar
from imipnet.datasets import sequence


class HPatchesPair(planar.HomographyPair):
    """Represents a stereo pair from HPatches"""

    def __init__(self: 'HPatchesPair', image_1: np.ndarray, image_2: np.ndarray, homography: np.ndarray,
                 seq_name: str, indices: Tuple[int, int]):
        name = "{0}: {1} {2}".format(seq_name, indices[0] + 1, indices[1] + 1)
        super().__init__(image_1, image_2, homography, name)


class HPatchesPairGenerator:
    """Generates pairs of stereo images from an HPatches Sequence"""

    def __init__(self, name: str, images: List[np.ndarray], homographies: List[np.ndarray]):
        self.name = name
        self.images = images
        self._homographies = homographies
        assert (len(self._homographies) == 5)
        self._pairs_order = [(i, j) for i in range(0, 6) for j in range(i + 1, 6)]

    @staticmethod
    def read_raw_folder(path: str, convert_to_grayscale: Optional[bool] = True) -> 'HPatchesPairGenerator':
        name = os.path.basename(path)
        images: List[np.ndarray] = list(iter(
            sequence.GlobImageSequence(
                os.path.join(path, "*.ppm"),
                load_as_grayscale=convert_to_grayscale
            )
        ))
        homography_paths = [os.path.join(path, "H_1_{0}".format(i)) for i in range(2, 7)]
        homographies: List[np.ndarray] = [np.loadtxt(homography_path) for homography_path in homography_paths]
        return HPatchesPairGenerator(name, images, homographies)

    def downsample_in_place(self: 'HPatchesPairGenerator') -> None:
        for i in range(len(self.images)):
            self.images[i] = cv2.pyrDown(self.images[i])

        h_half = np.array([[.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        h_double = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

        for i in range(len(self._homographies)):
            self._homographies[i] = np.dot(
                h_half,
                np.dot(
                    self._homographies[i],
                    h_double,
                )
            )

    def _get_homography(self: 'HPatchesPairGenerator', index_1: int, index_2: int) -> np.ndarray:
        # Subtract 1 from indices since the first homography (linear index == 0) is (0, 1)
        if index_1 == 0:
            return self._homographies[index_2 - 1]  # hom. (0 -> index_2)
        else:
            return np.dot(
                self._homographies[index_2 - 1],  # hom. (0 -> index_2)
                np.linalg.inv(self._homographies[index_1 - 1])  # hom. (index_1 -> 0)
            )

    def __len__(self: 'HPatchesPairGenerator') -> int:
        return len(self._pairs_order)

    def __getitem__(self: 'HPatchesPairGenerator', linear_index: int) -> HPatchesPair:
        two_dim_index = self._pairs_order[linear_index]
        image_1 = self.images[two_dim_index[0]]
        image_2 = self.images[two_dim_index[1]]
        homography = self._get_homography(two_dim_index[0], two_dim_index[1])
        return HPatchesPair(image_1, image_2, homography, self.name, two_dim_index)


class HPatchesSequencesStereoPairs(torch.utils.data.Dataset):
    """Loads the HPatches sequences train_dataset and returns stereo pairs of images from each kitti_sequence"""

    url: str = 'http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz'
    pairs_per_sequence: int = 15  # 6 choose 2 pairs

    train_sequences: List[str] = ['i_crownnight', 'i_dc', 'i_dome', 'i_duda', 'i_fenis', 'i_gonnenberg', 'i_greenhouse',
                                  'i_greentea', 'i_indiana', 'i_kions', 'i_ktirio', 'i_leuven', 'i_lionday',
                                  'i_londonbridge', 'i_melon', 'i_miniature', 'i_nescafe', 'i_nuts', 'i_objects',
                                  'i_parking', 'i_partyfood', 'i_pencils', 'i_pinard', 'i_pool', 'i_school', 'i_ski',
                                  'i_smurf', 'i_steps', 'i_toy', 'i_veggies', 'i_village', 'i_yellowtent', 'v_beyus',
                                  'v_bip', 'v_bird', 'v_blueprint', 'v_boat', 'v_bricks', 'v_calder', 'v_cartooncity',
                                  'v_charing', 'v_churchill', 'v_circus', 'v_colors', 'v_dirtywall', 'v_dogman',
                                  'v_gardens', 'v_grace', 'v_graffiti', 'v_home', 'v_laptop', 'v_london', 'v_machines',
                                  'v_maskedman', 'v_posters', 'v_samples', 'v_sunseason', 'v_tempera', 'v_there',
                                  'v_vitro', 'v_wall', 'v_wapping', 'v_war', 'v_weapons', 'v_wormhole', 'v_wounded']

    validation_sequences: List[str] = ['i_boutique', 'i_castle', 'i_chestnuts', 'i_contruction', 'i_crownday', 'v_adam',
                                       'v_apprentices', 'v_artisans', 'v_astronautis', 'v_bark']

    test_sequences: List[str] = ["i_ajuntament", "i_resort", "i_table", "i_troulos", "i_bologna", "i_lionnight",
                                 "i_porta", "i_zion", "i_brooklyn", "i_fruits", "i_books", "i_bridger",
                                 "i_whitebuilding", "i_kurhaus", "i_salon", "i_autannes", "i_tools", "i_santuario",
                                 "i_fog", "i_nijmegen", "v_courses", "v_coffeehouse", "v_abstract", "v_feast",
                                 "v_woman", "v_talent", "v_tabletop", "v_bees", "v_strand", "v_fest", "v_yard",
                                 "v_underground", "v_azzola", "v_eastsouth", "v_yuri", "v_soldiers", "v_man",
                                 "v_pomegranate", "v_birdwoman", "v_busstop"]

    @property
    def raw_folder(self: 'HPatchesSequencesStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'raw')

    @property
    def raw_extracted_folder(self: 'HPatchesSequencesStereoPairs') -> str:
        return os.path.join(self.raw_folder, "hpatches-sequences-release")

    @property
    def processed_folder(self: 'HPatchesSequencesStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'processed')

    @property
    def processed_file(self: 'HPatchesSequencesStereoPairs') -> str:
        file_name = "hpatches-"
        if self.downsample_large_images:
            file_name += "downsampled-"
        if self.convert_to_grayscale:
            file_name += "grayscale-"
        if self.require_pose_changes:
            file_name += "view-change-"
        if self.train:
            file_name += "training-"
        else:
            file_name += "testing-"
        file_name += "sequences.pickle"

        return os.path.join(self.processed_folder, file_name)

    def __init__(self: 'HPatchesSequencesStereoPairs', root: str,
                 split: Optional[str] = None,
                 download: Optional[bool] = False,
                 require_pose_changes: Optional[bool] = True,
                 downsample_large_images: Optional[bool] = True,
                 convert_to_grayscale: Optional[bool] = True) -> None:
        self.root_folder = root
        self.require_pose_changes = require_pose_changes
        self.downsample_large_images = downsample_large_images
        self.convert_to_grayscale = convert_to_grayscale

        if split == "train" or split is None:
            split_sequences = HPatchesSequencesStereoPairs.train_sequences
        elif split == "validation":
            split_sequences = HPatchesSequencesStereoPairs.validation_sequences
        elif split == "test":
            split_sequences = HPatchesSequencesStereoPairs.test_sequences
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")

        if self.require_pose_changes:
            split_sequences = [seq_folder for seq_folder in split_sequences if seq_folder.startswith('v_')]

        self._split_sequences = split_sequences

        if download:
            self.download()

        if not self._check_processed_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        with open(self.processed_file, 'rb') as pickle_file:
            self.sequences = pickle.load(pickle_file)
        return

    def __len__(self: 'HPatchesSequencesStereoPairs') -> int:
        return self.pairs_per_sequence * len(self.sequences)

    def __getitem__(self: 'HPatchesSequencesStereoPairs', index: int) -> HPatchesPair:
        return self.sequences[index // self.pairs_per_sequence][index % self.pairs_per_sequence]

    def download(self: 'HPatchesSequencesStereoPairs') -> None:

        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)

            tv_data.download_and_extract_archive(
                HPatchesSequencesStereoPairs.url, download_root=self.raw_folder, remove_finished=True
            )

        if not self._check_processed_exists():
            hpatches_folders = [
                os.path.join(self.raw_extracted_folder, seq_folder) for seq_folder in self._split_sequences
            ]

            sequences = [
                HPatchesPairGenerator.read_raw_folder(seq_folder, self.convert_to_grayscale) for seq_folder in
                hpatches_folders
            ]

            # IMIPS downscales larges images
            if self.downsample_large_images:
                for seq in sequences:
                    while np.any(np.array(seq.images[0].shape) > 1000):
                        seq.downsample_in_place()

            os.makedirs(self.processed_folder, exist_ok=True)
            with open(self.processed_file, 'wb') as pickle_file:
                pickle.dump(sequences, pickle_file)

    def _check_raw_exists(self: 'HPatchesSequencesStereoPairs') -> bool:
        return os.path.exists(self.raw_extracted_folder)

    def _check_processed_exists(self: 'HPatchesSequencesStereoPairs') -> bool:
        return os.path.exists(self.processed_file)
