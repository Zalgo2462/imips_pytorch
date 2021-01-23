import timeit
from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy as np
import torch

from imipnet.data.image import load_image_for_torch
from imipnet.lightning_module import test_dataset_registry, IMIPLightning


class SIFT:
    def __init__(self):
        self.sift_handle = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    # kitti-gray-0.5[0] 117s / 1000
    def correspondences_np(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        keypoints = []
        features = []
        for image in images:
            img_keypoints, img_features = self.sift_handle.detectAndCompute(image, None)
            img_resort = np.argsort(-1 * np.array([x.response for x in img_keypoints]))
            img_keypoints = [img_keypoints[i] for i in img_resort][:128]
            img_features = img_features[img_resort, :][:128, :]
            keypoints.append(img_keypoints)
            features.append(img_features)

        # naive multi-image feature matching
        anchor_features = features[0]
        matches = []
        for other_features in features[1:]:
            matches.append(self.bf_matcher.match(anchor_features, other_features))

        matched_kps = []
        for i in range(len(matches)):
            matched_kps.append((
                np.array([keypoints[0][match.queryIdx].pt for match in matches[i]]).transpose(),
                np.array([keypoints[i + 1][match.trainIdx].pt for match in matches[i]]).transpose()
            ))
        return matched_kps


class IMIPNet:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device

        checkpoint_net = IMIPLightning.load_from_checkpoint(checkpoint_path, strict=False)  # calls seed everything
        checkpoint_net.freeze()
        self.network = checkpoint_net.network.to(device=self.device)

    # kitti-gray-0.5[0] 304s / 1000
    def correspondences_np(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        image_batch = torch.stack([load_image_for_torch(img, device=self.device) for img in images], dim=0)
        return self.correspondences_torch(image_batch)

    # kitti-gray-0.5[0] 304s / 1000
    def correspondences_torch(self, image_batch: torch.tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
        batched_corrs = self.network.extract_keypoints_batched(image_batch)[0].cpu()
        matched_kps = []
        for i in range(image_batch.shape[0] - 1):
            matched_kps.append((
                batched_corrs[0].numpy(),
                batched_corrs[i + 1].numpy()
            ))
        return matched_kps


def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument('test_set', choices=test_dataset_registry.keys())
    parser.add_argument('--data_root', default="./data")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    params = parser.parse_args()

    test_set = test_dataset_registry[params.test_set](params.data_root)

    pair = test_set[0]
    # images = [pair.image_1, pair.image_2]
    images = [pair.image_1] * 16

    sift_corr_engine = SIFT()
    imip_corr_engine = IMIPNet(checkpoint_path=params.checkpoint)

    print("Loaded")

    sift_seconds = timeit.timeit(lambda: sift_corr_engine.correspondences_np(images), number=1)

    image_batch = torch.stack([load_image_for_torch(img, device=imip_corr_engine.device) for img in images], dim=0)
    imip_seconds = timeit.timeit(lambda: imip_corr_engine.correspondences_torch(image_batch), number=1)

    print("SIFT: %f" % sift_seconds)
    print("IMIP: %f" % imip_seconds)
    return


if __name__ == '__main__':
    main()
