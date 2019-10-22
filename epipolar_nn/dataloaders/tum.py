import os
from typing import Optional, Iterable

import docker
import numpy as np
import torch.utils.data


class VideoStereoPairGenerator:

    def __init__(self: 'VideoStereoPairGenerator', name: str, images: Iterable[np.ndarray], klt_tracker):
        self.name = name
        self.images = images
        self._klt_tracks = klt_tracker.find_tracks_in_sequence(self.images)


class TUMMonocularStereoPairs(torch.utils.data.Dataset):
    sequence_names = ["sequence_{0:02d}".format(i) for i in range(1, 51)]

    @property
    def raw_folder(self: 'TUMMonocularStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self: 'TUMMonocularStereoPairs') -> str:
        return os.path.join(self.root_folder, self.__class__.__name__, 'processed')

    def __init__(self: 'TUMMonocularStereoPairs', root: str,
                 download: Optional[bool] = False) -> None:
        self.root_folder = os.path.abspath(root)

        if download:
            self.download()

    def download(self: 'TUMMonocularStereoPairs') -> None:
        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            self._run_dockerized_tum_rectifier()
        # Processing/ Tracking goes here in a check_processed_exists clause

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
        for seq_name in TUMMonocularStereoPairs.sequence_names:
            if not os.path.exists(os.path.join(self.raw_folder, seq_name, "rect")):
                return False
        return True
