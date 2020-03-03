import cv2
import matplotlib.pylab as plt
import numpy as np
import scipy.stats

from epipointnet.data.calibrated import STD_STEREO_F_MAT_ALGORITHM
from epipointnet.data.pairs import CorrespondenceFundamentalMatrixPair
from epipointnet.datasets.tum_rgbd import TUMRGBDStereoPairs, TUMRGBDDataset

test_dataset_name = TUMRGBDDataset.FR3_DATASETS[0]
# For FR3_DATASETS[0], best_pair = 48062, worst_pair = 52485
# test_dataset_name = "fr1/xyz"
# For fr1/xyz, 195 has translation > 0.001 with little rotation

print("Loading Dataset...")
dataset = TUMRGBDStereoPairs("./data", test_dataset_name, download=True, f_matrix_algorithm=STD_STEREO_F_MAT_ALGORITHM)

FAST_detector = cv2.FastFeatureDetector_create()
l1_alg_errs = []

print("Displaying Images")
j = -1
for pair in dataset:
    j = j + 1
    pair: CorrespondenceFundamentalMatrixPair = pair
    if j % 100 == 0:
        print("")

    if np.any(pair._f_pair.baseline_1_2 < 0.0008 * 2) or pair._f_pair.rotation_deg < 0.5 * 2:
        print("s", end="", flush=True)
        continue

    """ Generate correspondences """
    image_1_sample_points_cv = FAST_detector.detect(pair.image_1)
    image_1_anchors = np.array([[i.pt[0], i.pt[1]] for i in image_1_sample_points_cv]).T
    image_1_sample_point_responses = np.array([i.response for i in image_1_sample_points_cv])
    image_1_anchors = image_1_anchors[:, np.argsort(image_1_sample_point_responses)]
    image_1_anchors = image_1_anchors[:, 0:min(image_1_anchors.shape[1], 50)]

    # image_1_anchors and image_2_correspondences are in image 1
    # image_2_anchors and image_1_correspondences are in image 2

    image_1_correspondences, image_1_anchor_points_idx = pair.correspondences(image_1_anchors, inverse=False)
    image_1_anchors = image_1_anchors[:, image_1_anchor_points_idx]

    image_2_correspondences, image_1_correspondences_idx = pair.correspondences(image_1_correspondences, inverse=True)
    image_1_correspondences = image_1_correspondences[:, image_1_correspondences_idx]
    image_1_anchors = image_1_anchors[:, image_1_correspondences_idx]

    inliers_idx = np.linalg.norm(image_1_anchors - image_2_correspondences, axis=0) < 1
    image_1_anchors = image_1_anchors[:, inliers_idx]
    image_1_correspondences = image_1_correspondences[:, inliers_idx]
    image_2_correspondences = image_2_correspondences[:, inliers_idx]

    if image_1_anchors.shape[1] == 0:
        print("!", end="", flush=True)
        continue

    """ Calculate algebraic error """

    l1_algebraic_error_pairs = np.abs(np.diag(
        np.vstack((image_1_correspondences, np.ones(image_1_correspondences.shape[1]))).T @
        pair.f_matrix_forward @
        np.vstack((image_1_anchors, np.ones(image_1_anchors.shape[1])))
    ))

    l1_algebraic_error = np.mean(l1_algebraic_error_pairs)
    l1_alg_errs.append([j, l1_algebraic_error])

    # """ FILTER DISPLAYED DATA """
    # if j % 100 != 0:
    #     print(".", end="", flush=True)
    #     continue

    print("")

    """ UPDATE THE ERROR DISPLAY """
    _, minmax, mean, variance, _, _ = scipy.stats.describe([x[1] for x in l1_alg_errs])
    print("All Pairs:\tL1 Algebraic Error:\tMin: {0:.2f};\tMax: {1:.2f};\tMean: {2:.2f};\tVariance {3:.2f}".format(
        minmax[0], minmax[1], mean, variance
    ))
    print(
        "Current Pair:\tIndex: {0};\tName: {1};\tL1 Algebraic Error: {2:.2f}".format(j, pair.name, l1_algebraic_error))

    """ SHOW IMAGES """
    _, axes = plt.subplots(1, 2, num=0)
    axes[0].imshow(pair.image_1)
    axes[1].imshow(pair.image_2)

    """ SHOW CORRESPONDENCES FROM 1 TO 2 """
    colors = np.random.rand(image_1_anchors.shape[1], 3)
    axes[0].scatter(image_1_anchors[0, :], image_1_anchors[1, :], c=colors, s=30)
    axes[1].scatter(image_1_correspondences[0, :], image_1_correspondences[1, :], c=colors, s=30)

    """ SHOW EPIPOLES """
    u, s, v_t = np.linalg.svd(pair.f_matrix_forward)
    epi_1 = v_t.T[:, -1]
    epi_1 = epi_1 / epi_1[-1]
    epi_2 = u[:, -1]
    epi_2 = epi_2 / epi_2[-1]

    axes[0].scatter([epi_1[0]], [epi_1[1]], c='g', s=60)
    axes[1].scatter([epi_2[0]], [epi_2[1]], c='g', s=60)

    """ SHOW EPIPOLAR LINES """
    """
       x * x_1 + y * y_1 + 1 * c_1 = 0
       y = -1 * (x * x_1 + 1 * c_1) / y_1
       y = -1 * (x * x_1/y_1 + 1 * c_1/ y_1)
       y = -1 * (x * x_1/y_1 + 1 * c_1/ y_1)
    """

    img_1_lines = (np.vstack(
        (image_1_correspondences, np.ones(image_1_correspondences.shape[1]))).T @ pair.f_matrix_forward).T
    # 2xN -> 3xN -> Nx3 @ 3x3 -> Nx3 -> 3xN
    x_vals = (np.linspace(0, 1, num=100)[:, np.newaxis] * pair.image_1.shape[1]).repeat(img_1_lines.shape[1], axis=1)
    y_vals = -1 * (x_vals * img_1_lines[0, :] / img_1_lines[1, :] + img_1_lines[2, :] / img_1_lines[1, :])

    for i in range(img_1_lines.shape[1]):
        line_x_vals = x_vals[:, i]
        line_y_vals = y_vals[:, i]
        axes[0].plot(line_x_vals, line_y_vals, c=colors[i, :])

    img_2_lines = pair.f_matrix_forward @ np.vstack((image_1_anchors, np.ones(image_1_anchors.shape[1])))
    # 3x3 @ (2xN -> 3xN) -> 3xN
    x_vals = (np.linspace(0, 1, num=100)[:, np.newaxis] * pair.image_2.shape[1]).repeat(img_2_lines.shape[1], axis=1)
    y_vals = -1 * (x_vals * img_2_lines[0, :] / img_2_lines[1, :] + img_2_lines[2, :] / img_2_lines[1, :])

    for i in range(img_2_lines.shape[1]):
        line_x_vals = x_vals[:, i]
        line_y_vals = y_vals[:, i]
        axes[1].plot(line_x_vals, line_y_vals, c=colors[i, :])

    """ SET THE VIEW WINDOW """

    # axes[0].set_xlim(0, pair.image_1.shape[1])
    # axes[0].set_ylim(pair.image_1.shape[0], 0)
    #
    # axes[1].set_xlim(0, pair.image_2.shape[1])
    # axes[1].set_ylim(pair.image_2.shape[0], 0)

    """ DISPLAY THE CORRESPONDENCE PLOT """
    plt.show()
