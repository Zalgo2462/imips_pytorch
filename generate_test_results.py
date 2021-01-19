# factor list
# - center vs harris preprocessing ( 2 choices)
# - ohnm 1 vs 16 loss ( 2 choices )
# - train dataset modality ( 3 choices )
# - test dataset modality ( 3 choices )
# Overall choices: 2x2x3x3=36

import enum
import os
import runpy
import sys


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


test_dataset_param = {
    Dataset.FLOW: "kitti-gray-0.5",
    Dataset.MVS: "megadepth-gray",
    Dataset.SYNTHETIC: "blender-livingroom-gray",
}

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

snapshots = {
    (Preprocess.CENTER, Loss.STANDARD, Dataset.FLOW): os.path.join(
        experiment_dirs[Dataset.FLOW],
        "center-sc-14_ohnm-1_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-tum-mono_eval-kitti-gray-0.5_Jan08_20-57-43_daedalus"
    ),
    (Preprocess.CENTER, Loss.STANDARD, Dataset.MVS): os.path.join(
        experiment_dirs[Dataset.MVS],
        "center-sc-14_ohnm-1_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-megadepth-gray_eval-megadepth-gray_Nov21_19-29-42_daedalus"
    ),
    (Preprocess.CENTER, Loss.STANDARD, Dataset.SYNTHETIC): os.path.join(
        experiment_dirs[Dataset.SYNTHETIC],
        "center-sc-14_ohnm-1_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-blender-livingroom-gray_eval-blender-livingroom-gray_Dec07_21-35-24_ll-desktop"
    ),
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
    (Preprocess.HARRIS, Loss.STANDARD, Dataset.FLOW): os.path.join(
        experiment_dirs[Dataset.FLOW],
        "harris-sc-14_ohnm-1_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-tum-mono_eval-kitti-gray-0.5_Nov14_22-51-28_linux23"
    ),
    (Preprocess.HARRIS, Loss.STANDARD, Dataset.MVS): os.path.join(
        experiment_dirs[Dataset.MVS],
        "harris-sc-14_ohnm-1_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-megadepth-gray_eval-megadepth-gray_Nov18_17-06-16_ll-desktop"
    ),
    (Preprocess.HARRIS, Loss.STANDARD, Dataset.SYNTHETIC): os.path.join(
        experiment_dirs[Dataset.SYNTHETIC],
        "harris-sc-14_ohnm-1_simple-conv-model_outlier-balanced-bce-bce-uml-loss_train-blender-livingroom-gray_eval-blender-livingroom-gray_Jan08_20-59-02_linux23"
    ),
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

for checkpoint_dir in snapshots.values():
    try:
        assert os.path.exists(checkpoint_dir)
    except AssertionError:
        raise FileNotFoundError(checkpoint_dir)

for key, checkpoint_dir in snapshots.items():
    preprocess, loss, dataset = key

    checkpoints = os.listdir(checkpoint_dir)
    checkpoint = os.path.join(
        checkpoint_dir, [x for x in checkpoints if x.startswith("epoch=") and x.endswith(".ckpt")][0]
    )

    for test_set in Dataset:
        args = [
            "--data_root", "data",
            "--output_dir", "test_results",
            checkpoint, test_dataset_param[test_set]
        ]
        sys.argv = [sys.argv[0]] + args
        runpy.run_path(os.path.join("imipnet", "lightning_test.py"))
