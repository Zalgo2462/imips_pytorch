import os
import timeit
from argparse import ArgumentParser
from typing import List, Tuple

import cv2
import numpy as np
import torch
import tqdm

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
        self.network = checkpoint_net.to(device=self.device)

    def correspondences_np(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        image_batch = [load_image_for_torch(img, device=self.device) for img in images]
        return self.correspondences_torch(image_batch)

    def correspondences_torch(self, images: List[torch.tensor]) -> List[Tuple[np.ndarray, np.ndarray]]:
        images = [self.network.preprocess(img) for img in images]
        same_size = True
        shape1 = images[0].shape
        for image in images[1:]:
            if image.shape != shape1:
                same_size = False
                break
        if same_size:
            image_batch = torch.stack(images, dim=0)
            batched_corrs = self.network.network.extract_keypoints_batched(image_batch)[0].cpu()
        else:
            batched_corrs = torch.stack(
                [self.network.network.extract_keypoints(image)[0].cpu() for image in images],
                dim=0
            )
        matched_kps = []
        for i in range(len(images) // 2):
            matched_kps.append((
                batched_corrs[0].numpy(),
                batched_corrs[i + 1].numpy()
            ))
        return matched_kps


def main():
    parser = ArgumentParser()
    parser.add_argument('--test_set', choices=test_dataset_registry.keys(), default="kitti-gray-0.5")
    parser.add_argument('--data_root', default="./data")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    params = parser.parse_args()

    output_subdir = os.path.join(params.output_dir, params.test_set, "all")
    os.makedirs(output_subdir, exist_ok=True)

    test_set = test_dataset_registry[params.test_set](params.data_root)
    print("Loaded test dataset")

    flow_checkpoints_dir = "./experiments/Lightning -- TUM 2 KITTI OL=0.5/checkpoints/simple-conv"
    center_ohnm16_chkpt_path = os.path.join(
        flow_checkpoints_dir,
        "center-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-tum-mono_eval-kitti-gray-0.5_Jan08_20-57-46_daedalus/epoch=1.ckpt"
    )
    harris_ohnm16_chkpt_path = os.path.join(
        flow_checkpoints_dir,
        "harris-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-tum-mono_eval-kitti-gray-0.5_Nov14_22-40-40_daedalus/epoch=2_v0.ckpt"
    )

    center_imipnet = IMIPNet(checkpoint_path=center_ohnm16_chkpt_path, device="cuda:0")
    harris_imipnet = IMIPNet(checkpoint_path=harris_ohnm16_chkpt_path, device="cuda:1")
    sift_corr = SIFT()
    print("Loaded correspondence engines")

    center_imipnet_timings = []
    harris_imipnet_timings = []
    sift_timings = []

    for pair in tqdm.tqdm(test_set, desc="Test Stereo Pairs"):
        images = [pair.image_1, pair.image_2]
        center_imipnet_seconds = timeit.timeit(lambda: center_imipnet.correspondences_np(images), number=1)
        harris_imipnet_seconds = timeit.timeit(lambda: harris_imipnet.correspondences_np(images), number=1)
        sift_seconds = timeit.timeit(lambda: sift_corr.correspondences_np(images), number=1)
        center_imipnet_timings.append(center_imipnet_seconds)
        harris_imipnet_timings.append(harris_imipnet_seconds)
        sift_timings.append(sift_seconds)

    torch.save(torch.tensor(center_imipnet_timings), os.path.join(output_subdir, "center_imipnet_timings.pt"))
    torch.save(torch.tensor(harris_imipnet_timings), os.path.join(output_subdir, "harris_imipnet_timings.pt"))
    torch.save(torch.tensor(sift_timings), os.path.join(output_subdir, "sift_timings.pt"))
    return


if __name__ == '__main__':
    main()
