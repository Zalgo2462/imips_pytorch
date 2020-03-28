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


NormalizedEightPointNet with subnets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from epipointnet.dfe import loss as L


class WeightEstimatorNet(nn.Module):
    """Network for weight estimation.
    """

    def __init__(self, input_size, inplace=True, has_bias=True, learn_affine=True):
        """Init.

        Args:
            input_size (int): size of input
            inplace (bool, optional): Defaults to True. LeakyReLU inplace?
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """

        super(WeightEstimatorNet, self).__init__()

        track = False  # TODO: try turning this on since I generally train with batch size = 1
        self.model = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(64, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(64, 128, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(128, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(128, 1024, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(1024, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(1024, 512, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(512, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(512, 256, kernel_size=1, bias=has_bias),
            nn.InstanceNorm1d(256, affine=learn_affine, track_running_stats=track),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv1d(256, 1, kernel_size=1, bias=has_bias),
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data (tensor): input data

        Returns:
            tensor: forward pass
        """

        return self.model(data)


class RescaleAndExpand(nn.Module):
    """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.
    """

    def __init__(self):
        """Init.
        """

        super(RescaleAndExpand, self).__init__()

        self.register_buffer("ones", torch.ones((1, 1, 1)))

    def normalize(self, pts):
        """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.

        Args:
            pts (tensor): input points

        Returns:
            tensor: transformed points
            tensor: transformation
        """

        ones = self.ones.expand(pts.size(0), pts.size(1), 1)

        pts = torch.cat((pts, ones), 2)

        center = torch.mean(pts, 1)
        dist = pts - center.unsqueeze(1)

        meandist = dist[:, :, :2].norm(p=2, dim=2).mean(1)

        scale = 1.0 / meandist

        transform = torch.zeros((pts.size(0), 3, 3), device=pts.device)

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = torch.bmm(transform, pts.permute(0, 2, 1))

        return pts_out, transform

    def forward(self, pts):
        """Forward pass.

        Args:
            pts (tensor): point correspondences

        Returns:
            tensor: transformed points in first image
            tensor: transformed points in second image
            tensor: transformtion (first image)
            tensor: transformtion (second image)
        """

        pts1, transform1 = self.normalize(pts[:, :, :2])
        pts2, transform2 = self.normalize(pts[:, :, 2:])

        return pts1, pts2, transform1, transform2


class ModelEstimator(nn.Module):
    """Esimator for model.
    """

    def __init__(self):
        """Init.
        """

        super(ModelEstimator, self).__init__()

        self.register_buffer("mask", torch.ones(3))
        self.mask[-1] = 0

    def normalize(self, pts, weights):
        """Normalize points based on weights.

        Args:
            pts (tensor): points
            weights (tensor): estimated weights

        Returns:
            tensor: normalized points
        """

        denom = weights.sum(1)

        center = torch.sum(pts * weights, 1) / denom
        dist = pts - center.unsqueeze(1)

        meandist = (
                (weights * (dist[:, :, :2].norm(p=2, dim=2).unsqueeze(2))).sum(1)
                / denom
        ).squeeze(1)

        scale = 1.4142 / meandist

        transform = torch.zeros((pts.size(0), 3, 3), device=pts.device)

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = torch.bmm(transform, pts.permute(0, 2, 1))

        return pts_out, transform

    def weighted_svd(self, pts1, pts2, weights):
        """Solve homogeneous least squares problem and extract model.

        Args:
            pts1 (tensor): points in first image
            pts2 (tensor): points in second image
            weights (tensor): estimated weights

        Returns:
            tensor: estimated fundamental matrix
        """

        weights = weights.squeeze(1).unsqueeze(2)

        pts1n, transform1 = self.normalize(pts1, weights)
        pts2n, transform2 = self.normalize(pts2, weights)

        p = torch.cat(
            (pts1n[:, 0].unsqueeze(1) * pts2n, pts1n[:, 1].unsqueeze(1) * pts2n, pts2n),
            1,
        ).permute(0, 2, 1)

        X = p * weights

        out_batch = []

        for batch in range(X.size(0)):
            # solve homogeneous least squares problem
            # TODO: If SVD nans are encountered try zeroing them out with
            #  [batch].register_hook(lambda grad: torch.zeros_like(grad) if torch.isnan(grad) else grad)
            _, _, V = torch.svd(X[batch])
            F = V[:, -1].view(3, 3)

            # model extractor
            # TODO: If SVD nans are encountered try zeroing them out with
            #  F.register_hook(lambda grad: torch.zeros_like(grad) if torch.isnan(grad) else grad)
            U, S, V = torch.svd(F)
            F_projected = U.mm((S * self.mask).diag()).mm(V.t())

            out_batch.append(F_projected.unsqueeze(0))

        out = torch.cat(out_batch, 0)
        out = transform1.permute(0, 2, 1).bmm(out).bmm(transform2)

        return out

    def forward(self, pts1, pts2, weights):
        """Forward pass.

        Args:
            pts1 (tensor): points in first image
            pts2 (tensor): points in second image
            weights (tensor): estimated weights

        Returns:
            tensor: estimated fundamental matrix
        """

        out = self.weighted_svd(pts1, pts2, weights)

        return out


class NormalizedEightPointNet(nn.Module):
    """NormalizedEightPointNet for fundamental matrix estimation.

    The output of the forward pass is the fundamental matrix and the rescaling matrices that
    transform the input points to [-1, 1]^2.

    The input are the point correspondences as well as the associated  side information.
    """

    def __init__(self, depth=1, side_info_size=0):
        """Init.
            depth (int, optional): Defaults to 1. [description]
            side_info_size (int, optional): Defaults to 0. [description]
        """

        super(NormalizedEightPointNet, self).__init__()

        self.depth = depth

        # data processing
        self.rescale_and_expand = RescaleAndExpand()

        # model estimator
        self.model = ModelEstimator()

        # weight estimator
        self.side_info_size = side_info_size
        self.weights_init = WeightEstimatorNet(4 + side_info_size)
        self.weights_iter = WeightEstimatorNet(6 + side_info_size)

    def forward(self, pts, side_info):
        """Forward pass.

        Args:
            pts batch x corr_n x (pt1, pt2) (tensor): point correspondences
            side_info batch x corr_n x side_info (tensor): side information

        Returns:
            tensor: fundamental matrix, transformation of points in first and second image
        """

        # recale points to [-1, 1]^2 and expand with 1
        pts1, pts2, rescaling_1, rescaling_2 = self.rescale_and_expand(pts)

        pts1 = pts1.permute(0, 2, 1)
        pts2 = pts2.permute(0, 2, 1)

        # init weights
        input_p_s = torch.cat(
            ((pts1[:, :, :2] + 1) / 2, (pts2[:, :, :2] + 1) / 2, side_info), 2
        ).permute(0, 2, 1)
        weights = F.softmax(self.weights_init(input_p_s), dim=2)

        out_depth = self.model(pts1, pts2, weights)
        out = [out_depth]

        # iter weights
        for _ in range(1, self.depth):
            residual = L.robust_symmetric_epipolar_distance(pts1, pts2, out_depth)

            input_p_s_w_r = torch.cat((input_p_s, weights, residual.unsqueeze(1)), 1)

            weights = F.softmax(self.weights_iter(input_p_s_w_r), dim=2)
            out_depth = self.model(pts1, pts2, weights)
            out.append(out_depth)

        return out, rescaling_1, rescaling_2, weights
