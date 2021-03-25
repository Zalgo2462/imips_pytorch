import enum
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from imipnet.data.image import load_image_for_torch
from imipnet.lightning_module import preprocess_registry, test_dataset_registry, IMIPLightning


class Preprocess(enum.Enum):
    CENTER = 1
    HARRIS = 2


class Loss(enum.Enum):
    STANDARD = 1
    OHNM16 = 2


class Dataset(enum.Enum):
    FLOW = 1
    MVS = 2
    SYNTHETIC = 3


experiment_dirs = {
    Dataset.FLOW: os.path.realpath(os.path.join(
        "experiments", "Lightning -- TUM 2 KITTI OL=0.5", "checkpoints", "simple-conv"
    )),
    Dataset.MVS: os.path.realpath(os.path.join(
        "experiments", "Lightning -- MegaDepth", "checkpoints", "simple-conv"
    )),
    Dataset.SYNTHETIC: os.path.realpath(os.path.join(
        "experiments", "Lightning -- Blender", "checkpoints", "simple-conv"
    )),
}
"""
snapshots_dirs = {
    (Preprocess.HARRIS, Loss.OHNM16, Dataset.FLOW): os.path.join(
        experiment_dirs[Dataset.FLOW],
        "harris-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-tum-mono_eval-kitti-gray-0.5_Nov14_22-40-40_daedalus"
    ),
    (Preprocess.HARRIS, Loss.OHNM16, Dataset.MVS): os.path.join(
        experiment_dirs[Dataset.MVS],
        "harris-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-megadepth-gray_eval-megadepth-gray_Nov18_20-38-16_daedalus"
    ),
    (Preprocess.HARRIS, Loss.OHNM16, Dataset.SYNTHETIC): os.path.join(
        experiment_dirs[Dataset.SYNTHETIC],
        "harris-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-blender-livingroom-gray_eval-blender-livingroom-gray_Dec18_17-25-26_daedalus"
    ),
}

model_dataset_map = {
    (Preprocess.HARRIS, Loss.OHNM16, Dataset.FLOW): [
        lambda data_root: test_dataset_registry["tum-mono"](data_root),
        lambda data_root: test_dataset_registry["kitti-gray-0.5"](data_root)
    ],
    (Preprocess.HARRIS, Loss.OHNM16, Dataset.MVS): [
        lambda data_root: test_dataset_registry["megadepth-gray"](data_root),
    ],
    (Preprocess.HARRIS, Loss.OHNM16, Dataset.SYNTHETIC): [
        lambda data_root: test_dataset_registry["blender-livingroom-gray"](data_root),
    ],
}
"""

snapshots_dirs = {
    (Preprocess.CENTER, Loss.OHNM16, Dataset.FLOW): os.path.join(
        experiment_dirs[Dataset.FLOW],
        "center-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-tum-mono_eval-kitti-gray-0.5_Jan08_20-57-46_daedalus"
    ),
    (Preprocess.CENTER, Loss.OHNM16, Dataset.MVS): os.path.join(
        experiment_dirs[Dataset.MVS],
        "center-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-megadepth-gray_eval-megadepth-gray_Nov18_18-21-55_daedalus"
    ),
    (Preprocess.CENTER, Loss.OHNM16, Dataset.SYNTHETIC): os.path.join(
        experiment_dirs[Dataset.SYNTHETIC],
        "center-sc-14_ohnm-16_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-blender-livingroom-gray_eval-blender-livingroom-gray_Dec14_18-27-12_daedalus"
    ),
}

model_dataset_map = {
    (Preprocess.CENTER, Loss.OHNM16, Dataset.FLOW): [
        lambda data_root: test_dataset_registry["tum-mono"](data_root),
        lambda data_root: test_dataset_registry["kitti-gray-0.5"](data_root)
    ],
    (Preprocess.CENTER, Loss.OHNM16, Dataset.MVS): [
        lambda data_root: test_dataset_registry["megadepth-gray"](data_root),
    ],
    (Preprocess.CENTER, Loss.OHNM16, Dataset.SYNTHETIC): [
        lambda data_root: test_dataset_registry["blender-livingroom-gray"](data_root),
    ],
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_root', default="./data")
    parser.add_argument("--output_dir", type=str, default="./example_matches")
    params = parser.parse_args()

    preprocessor = preprocess_registry["center"]()

    os.makedirs(params.output_dir, exist_ok=True)

    for model_key, dataset_constructor_list in model_dataset_map.items():
        snapshot_dir = snapshots_dirs[model_key]
        checkpoints = os.listdir(snapshot_dir)
        checkpoint_path = os.path.join(
            snapshot_dir, [x for x in checkpoints if x.startswith("epoch=") and x.endswith(".ckpt")][0]
        )
        checkpoint_net = IMIPLightning.load_from_checkpoint(checkpoint_path, strict=False)  # calls seed everything
        checkpoint_net = checkpoint_net.to(device="cuda")
        checkpoint_net.freeze()

        for dataset_constructor in dataset_constructor_list:
            dataset = dataset_constructor(params.data_root)
            for i in range(100):
                pair = dataset[np.random.randint(0, len(dataset))]

                image_1_torch = preprocessor(load_image_for_torch(pair.image_1, device="cuda"))
                image_2_torch = preprocessor(load_image_for_torch(pair.image_2, device="cuda"))
                image_1_corrs = checkpoint_net.network.extract_keypoints(image_1_torch)[0]
                image_2_corrs = checkpoint_net.network.extract_keypoints(image_2_torch)[0]
                batched_corrs = torch.stack((image_1_corrs, image_2_corrs), dim=0).cpu()

                # Filter out invalid correspondences
                ground_truth_corrs, tracked_idx = pair.correspondences(batched_corrs[0].numpy(), inverse=False)
                batched_corrs = batched_corrs[:, :, tracked_idx]
                valid_idx = np.linalg.norm(ground_truth_corrs - batched_corrs[1].numpy(), ord=2, axis=0) < 1
                batched_corrs = batched_corrs[:, :, valid_idx]

                batched_corrs = batched_corrs.round()
                anchor_keypoints = [cv2.KeyPoint(int(batched_corrs[0][0][i]), int(batched_corrs[0][1][i]), 1) for i in
                                    range(batched_corrs.shape[2])]
                corr_keypoints = [cv2.KeyPoint(int(batched_corrs[1][0][i]), int(batched_corrs[1][1][i]), 1) for i in
                                  range(batched_corrs.shape[2])]
                matches = [cv2.DMatch(i, i, 0.0) for i in range(len(anchor_keypoints))]

                image_1 = cv2.cvtColor(pair.image_1, cv2.COLOR_RGB2BGR)
                image_2 = cv2.cvtColor(pair.image_2, cv2.COLOR_RGB2BGR)

                match_img_1 = cv2.drawMatches(image_1, anchor_keypoints, image_2, corr_keypoints, matches,
                                              None)
                pair_name = checkpoint_net.get_name() + "_" + pair.name.replace("/", "_") + ".png"
                cv2.imwrite(os.path.join(params.output_dir, pair_name), match_img_1)
                print("Wrote {0}".format(os.path.join(params.output_dir, pair_name)), flush=True)


if __name__ == '__main__':
    main()
