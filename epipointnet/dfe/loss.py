"""
MIT License

Copyright (c) 2018 Rene Ranftl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Loss functions.
"""
import torch


def symmetric_epipolar_distance(pts1, pts2, fundamental_mat):
    """Symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: symmetric epipolar distance
    """

    line_1 = torch.bmm(pts2, fundamental_mat)
    line_2 = torch.bmm(pts1, fundamental_mat.permute(0, 2, 1))

    # TODO: Add epsilon if unstable
    line_1_norm = line_1[:, :, :2].norm(2, 2)
    line_2_norm = line_2[:, :, :2].norm(2, 2)

    scalar_product = (pts1 * line_1).sum(2)

    ret = scalar_product.abs() * (
            1 / line_1_norm + 1 / line_2_norm
    )

    return ret


# def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=1.0):
def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=0.5):
    """Robust symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix
        gamma (float, optional): Defaults to 0.5. robust parameter

    Returns:
        tensor: robust symmetric epipolar distance
    """

    sed = symmetric_epipolar_distance(pts1, pts2, fundamental_mat)
    ret = torch.clamp(sed, max=gamma)

    return ret
