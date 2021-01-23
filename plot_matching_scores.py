import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_matching_scores(title, run_names, matching_score_arrays, line_styles):
    plt.clf()
    frac_gt = [1 - (np.arange(1, len(matching_scores) + 1) / len(matching_scores)) for matching_scores in
               matching_score_arrays]

    plt.title(title)
    for i in range(len(matching_score_arrays)):
        plt.step(matching_score_arrays[i], frac_gt[i], linestyle=line_styles[i], label=run_names[i])
    plt.axvline(x=10. / 128., label='10 inliers at 128', color='black')
    plt.grid()
    plt.legend()
    plt.xlabel('Matching score')
    plt.ylabel('Fraction of pairs with higher matching score')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def result_filter(result_name: str) -> bool:
    return True  # "harris" in result_name or "sift" in result_name


def linestyle_map(result_name: str) -> Union[Tuple, str]:
    linestyle_tuple = [
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    if "kitti" in result_name:
        return "dashed"
    elif "blender" in result_name:
        return "dotted"
    elif "megadepth" in result_name:
        return "dashdot"
    elif "sift" in result_name:
        return "solid"
    return linestyle_tuple[0]


def main():
    testing_dataset = "kitti-gray-0.5"
    result_dir = os.path.join(".", "test_results", testing_dataset, "all")
    # get the names of the results to plot
    result_names = [x for x in os.listdir(result_dir) if x.endswith(".pt") and result_filter(x)]
    # map the names of the results to different line styles
    line_styles = [linestyle_map(result_name) for result_name in result_names]
    # sort the line styles to figure out how to order the results
    reorder = np.argsort([x + y for (x, y) in zip(line_styles, result_names)])
    line_styles = [line_styles[i] for i in reorder]
    result_names = [result_names[i] for i in reorder]

    # load the results
    result_files = [os.path.join(result_dir, x) for x in result_names]
    results = [torch.load(path) for path in result_files]

    # extract the data
    apparent_matching_scores = [torch.sort(result["matching_scores"]["apparent"].flatten()).values for result in
                                results]
    true_matching_scores = [torch.sort(result["matching_scores"]["true"].flatten()).values for result in results]

    draw_matching_scores("Apparent Matching Scores for " + testing_dataset + " (Testing)", result_names,
                         apparent_matching_scores, line_styles)
    draw_matching_scores("Unique Matching Scores for " + testing_dataset + " (Testing)", result_names,
                         true_matching_scores, line_styles)


if __name__ == "__main__":
    main()
