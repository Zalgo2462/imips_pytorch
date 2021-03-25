import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

testing_dataset = "blender-livingroom-gray"


def result_filter(result_name: str) -> bool:
    return ("harris" in result_name and "ohnm-16_" in result_name) or ("sift" in result_name)


def linestyle_map(result_name: str) -> Union[Tuple, str]:
    # linestyle_tuple = [
    #     ('loosely dotted', (0, (1, 10))),
    #     ('dotted', (0, (1, 1))),
    #     ('densely dotted', (0, (1, 1))),
    #
    #     ('loosely dashed', (0, (5, 10))),
    #     ('dashed', (0, (5, 5))),
    #     ('densely dashed', (0, (5, 1))),
    #
    #     ('loosely dashdotted', (0, (3, 10, 1, 10))),
    #     ('dashdotted', (0, (3, 5, 1, 5))),
    #     ('densely dashdotted', (0, (3, 1, 1, 1))),
    #
    #     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    #     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    #     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    #
    # if "kitti" in result_name:
    #     return "dashed"
    # elif "blender" in result_name:
    #     return "dotted"
    # elif "megadepth" in result_name:
    #     return "dashdot"
    # elif "sift" in result_name:
    #     return "solid"

    return "solid"


def name_map(result_name: str) -> str:
    if "train-blender" in result_name:
        return "Blender"
    elif "train-tum" in result_name:
        return "TUM MonoVO"
    elif "train-megadepth" in result_name:
        return "Landmarks10k"
    elif "sift" in result_name:
        return "SIFT-128"
    return result_name


def draw_matching_scores(run_names, matching_score_arrays, line_styles):
    plt.clf()
    frac_gt = [1 - (np.arange(1, len(matching_scores) + 1) / len(matching_scores)) for matching_scores in
               matching_score_arrays]
    mean_matching_scores = np.array([x.mean() for x in matching_score_arrays])
    sort_idx = mean_matching_scores.argsort()

    for i in reversed(sort_idx):
        plt.step(
            matching_score_arrays[i], frac_gt[i], linestyle=line_styles[i],
            label=run_names[i] + ": {:.2f}".format(mean_matching_scores[i])
        )
    plt.axvline(x=10. / 128., label='10 inliers at 128', color='black')
    plt.grid()
    plt.legend()
    plt.xlabel('Matching score')
    plt.ylabel('Fraction of pairs with higher matching score')
    # plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def main():
    result_dir = os.path.join(".", "test_results", testing_dataset, "all")
    # get the names of the results to plot
    result_names = [x for x in os.listdir(result_dir) if x.endswith(".pt") and result_filter(x)]
    pretty_names = [name_map(x) for x in result_names]
    # map the names of the results to different line styles
    line_styles = [linestyle_map(result_name) for result_name in result_names]

    # load the results
    result_files = [os.path.join(result_dir, x) for x in result_names]
    results = [torch.load(path) for path in result_files]

    # extract the data
    apparent_matching_scores = [torch.sort(result["matching_scores"]["apparent"].flatten()).values for result in
                                results]
    true_matching_scores = [torch.sort(result["matching_scores"]["true"].flatten()).values for result in results]

    draw_matching_scores(pretty_names, apparent_matching_scores, line_styles)
    draw_matching_scores(pretty_names, true_matching_scores, line_styles)


if __name__ == "__main__":
    main()
