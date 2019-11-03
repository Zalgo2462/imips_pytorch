import os
import pickle
import shutil
from typing import Optional, Sequence, List

import docker
import numpy as np
import torch.utils.data

from . import klt, sequence, pair


class VideoStereoPairGenerator:

    def __init__(self: 'VideoStereoPairGenerator', name: str, images: Sequence[np.ndarray],
                 sequence_overlap: klt.SequenceOverlap):
        self.name = name
        self.img_sequence = images
        self.sequence_overlap = sequence_overlap

    def generate_random_stereo_pair(self: 'VideoStereoPairGenerator', minimum_overlap: float) -> pair.ImagePair:
        image_2_possibilies = np.array([])
        while image_2_possibilies.size == 0:
            image_1_index = np.random.randint(0, len(self.img_sequence))
            image_2_possibilies = self.sequence_overlap.find_frames_with_overlap(image_1_index, minimum_overlap)
        image_2_index = np.random.randint(0, image_2_possibilies.size)
        # TODO: Define PairWithIntermediates type ImagePair
        # TODO: Figure out how we go from random stereo pairs in old code to ordered stereo pairs in new code...
        raise NotImplementedError()


class TUMMonocularStereoPairs(torch.utils.data.Dataset):
    all_sequences = ["sequence_{0:02d}".format(i) for i in range(1, 51)]

    @property
    def train_sequences(self: 'TUMMonocularStereoPairs') -> List[str]:
        # IMIPS only trains on 1, 2, 3, 48, 49, 50 by default
        return [TUMMonocularStereoPairs.all_sequences[x] for x in [1, 2, 3, 48, 49, 50]]

    @property
    def test_sequences(self: 'TUMMonocularStereoPairs') -> List[str]:
        return [TUMMonocularStereoPairs.all_sequences[x] for x in range(4, 48)]

    @property
    def raw_folder(self: 'TUMMonocularStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self: 'TUMMonocularStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'processed')

    def __init__(self: 'TUMMonocularStereoPairs', root: str,
                 train: Optional[bool] = True,
                 download: Optional[bool] = False) -> None:
        self.root_folder = os.path.abspath(root)
        self.train = train

        if download:
            self.download()

        if not self._check_processed_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        sequence_names = self.train_sequences if self.train \
            else self.test_sequences

        self.stereo_pair_generators = []
        for seq_name in sequence_names:
            seq_path = os.path.join(self.processed_folder, seq_name)
            img_seq = sequence.GlobImageSequence(os.path.join(
                seq_path, "images", "*.jpg"
            ))
            with open(os.path.join(seq_path, "overlap.pickle"), 'rb') as overlap_file:
                seq_overlap = pickle.load(overlap_file)
            self.stereo_pair_generators.append(VideoStereoPairGenerator(seq_path, img_seq, seq_overlap))

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> pair.ImagePair:
        raise NotImplementedError()

    def download(self: 'TUMMonocularStereoPairs') -> None:
        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            self._run_dockerized_tum_rectifier()

        if not self._check_processed_exists():
            tracker = klt.Tracker()
            os.makedirs(self.processed_folder, exist_ok=True)
            for seq_name in self.all_sequences:
                old_seq_path = os.path.join(self.raw_folder, seq_name)
                new_seq_path = os.path.join(self.processed_folder, seq_name)
                old_seq_image_path = os.path.join(old_seq_path, "rect")
                new_seq_image_path = os.path.join(new_seq_path, "images")
                shutil.copytree(old_seq_image_path, new_seq_image_path)
                img_seq = sequence.GlobImageSequence(os.path.join(new_seq_image_path, "*.jpg"))
                seq_overlap = tracker.find_sequence_overlap(img_seq, max_num_points=500)
                with open(os.path.join(new_seq_path, "overlap.pickle"), 'wb') as overlap_file:
                    pickle.dump(seq_overlap, overlap_file)

    def _run_dockerized_tum_rectifier(self: 'TUMMonocularStereoPairs'):
        docker_client = docker.client.from_env()
        try:
            uid = os.getuid()
            gid = os.getgid()
        except AttributeError:
            uid = gid = 0

        build_streamer = docker_client.api.build(
            path='./tum_rectifier',
            tag='auto-tum-rectifier',
            decode=True,
            pull=True,
            buildargs={
                "UID": str(uid),
                "GID": str(gid),
            }
        )
        for chunk in build_streamer:
            if "stream" in chunk:
                print(chunk["stream"], end="")

        tum_rectifier = docker_client.containers.run(
            "auto-tum-rectifier",
            detach=True,
            auto_remove=True,
            volumes={
                self.raw_folder: {"bind": "/root/data", "mode": "rw"}
            }
        )
        tum_rectifier_logs = tum_rectifier.logs(stream=True, follow=True)
        for log in tum_rectifier_logs:
            print(log.decode("utf-8"), end="")

    def _check_raw_exists(self: 'TUMMonocularStereoPairs') -> bool:
        for seq_name in TUMMonocularStereoPairs.all_sequences:
            if not os.path.exists(os.path.join(self.raw_folder, seq_name, "rect")):
                return False
        return True

    def _check_processed_exists(self: 'TUMMonocularStereoPairs') -> bool:
        for seq_name in TUMMonocularStereoPairs.all_sequences:
            if not os.path.exists(os.path.join(self.processed_folder, seq_name, "images")) or not os.path.exists(
                    os.path.join(self.processed_folder, seq_name, "overlap.pickle")):
                return False
        return True
