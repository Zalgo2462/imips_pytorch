import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything

from imipnet.data.pairs import CorrespondencePair
from imipnet.datasets.shuffle import ShuffledDataset
from imipnet.lightning_module import test_dataset_registry

inlier_radius = 3
n_points = 128

parser = ArgumentParser()
parser.add_argument('test_set', choices=test_dataset_registry.keys())
parser.add_argument('--data_root', default="./data")
parser.add_argument('--n_eval_samples', type=int, default=-1)
parser.add_argument("--output_dir", type=str, default="./test_results")
parser.add_argument("--seed", type=int, default=0)
params = parser.parse_args()

seed_everything(params.seed)

eval_samples = None if params.n_eval_samples < 1 else params.n_eval_samples
test_set = ShuffledDataset(
    test_dataset_registry[params.test_set](params.data_root), eval_samples
)

# sift_handle = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.09)
sift_handle = cv2.SIFT_create()
bf_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

apparent_matching_scores = []
true_matching_scores = []

for pair in test_set:  # type: CorrespondencePair
    img_1_keypoints, img_1_features = sift_handle.detectAndCompute(pair.image_1, None)
    img_1_resort = np.argsort(-1 * np.array([x.response for x in img_1_keypoints]))
    img_1_keypoints = [img_1_keypoints[i] for i in img_1_resort][:128]
    img_1_features = img_1_features[img_1_resort, :][:128, :]

    img_2_keypoints, img_2_features = sift_handle.detectAndCompute(pair.image_2, None)
    img_2_resort = np.argsort(-1 * np.array([x.response for x in img_2_keypoints]))
    img_2_keypoints = [img_2_keypoints[i] for i in img_2_resort][:128]
    img_2_features = img_2_features[img_2_resort, :][:128, :]

    matches = bf_matcher.match(img_1_features, img_2_features)

    img_1_match_kps = np.array([img_1_keypoints[match.queryIdx].pt for match in matches]).transpose()
    img_2_match_kps = np.array([img_2_keypoints[match.trainIdx].pt for match in matches]).transpose()

    img_2_corrs_packed, img_2_corrs_idx = pair.correspondences(img_1_match_kps, inverse=False)
    img_2_corrs_unpacked = np.zeros_like(img_1_match_kps)
    img_2_corrs_unpacked[:, img_2_corrs_idx] = img_2_corrs_packed
    img_2_corrs_mask = np.zeros(img_2_corrs_unpacked.shape[1], dtype=np.bool)
    img_2_corrs_mask[img_2_corrs_idx] = True

    img_1_corrs_packed, img_1_corrs_idx = pair.correspondences(img_2_match_kps, inverse=True)
    img_1_corrs_unpacked = np.zeros_like(img_2_match_kps)
    img_1_corrs_unpacked[:, img_1_corrs_idx] = img_1_corrs_packed
    img_1_corrs_mask = np.zeros(img_1_corrs_unpacked.shape[1], dtype=np.bool)
    img_1_corrs_mask[img_1_corrs_idx] = True

    # TODO: convert corrs idx to mask

    img_1_inliers = (np.linalg.norm(img_1_match_kps - img_2_corrs_unpacked, ord=2,
                                    axis=0) < inlier_radius) & img_2_corrs_mask
    img_2_inliers = (np.linalg.norm(img_2_match_kps - img_1_corrs_unpacked, ord=2,
                                    axis=0) < inlier_radius) & img_1_corrs_mask

    apparent_inliers = (img_1_inliers & img_2_inliers)
    apparent_inliers_img_1_kp = img_1_match_kps[:, apparent_inliers]
    uniq_inliers_img_1_kp = apparent_inliers_img_1_kp[:, 0:1]
    for i in range(1, apparent_inliers_img_1_kp.shape[1]):
        test_inlier = apparent_inliers_img_1_kp[:, i:i + 1]
        if np.all(np.linalg.norm(uniq_inliers_img_1_kp - test_inlier, ord=2, axis=0) > inlier_radius):
            uniq_inliers_img_1_kp = np.hstack((uniq_inliers_img_1_kp, test_inlier))

    apparent_matching_scores.append(apparent_inliers_img_1_kp.shape[1] / n_points)
    true_matching_scores.append(uniq_inliers_img_1_kp.shape[1] / n_points)

result_dict = {
    "matching_scores": {
        "apparent": torch.tensor(apparent_matching_scores),
        "true": torch.tensor(true_matching_scores),
    }
}

if params.n_eval_samples < 1:
    n_eval_samples_str = "all"
else:
    n_eval_samples_str = str(params.n_eval_samples)

output_subdir = os.path.join(params.output_dir, params.test_set, n_eval_samples_str)
os.makedirs(output_subdir, exist_ok=True)

torch.save(result_dict, os.path.join(output_subdir, "sift-baseline" + ".pt"))
