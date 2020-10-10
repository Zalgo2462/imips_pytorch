from typing import Dict, List, Tuple

import torch
from kornia.filters import GaussianBlur2d

from epipointnet2.r2d2.modules.naver.patchnet import *


def load_r2d2_net(model_path: str) -> torch.nn.Module:
    checkpoint = torch.load(model_path)
    net = eval(checkpoint['net'])  # type: torch.nn.Module

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net


def freeze_module(module: torch.nn.Module):
    for p in module.parameters(True):
        p.requires_grad_(False)
    return


class NonDiffCorrespondenceEngine(torch.nn.Module):

    def __init__(self, r2d2_net: torch.nn.Module):
        super().__init__()
        self._r2d2_net = r2d2_net
        freeze_module(self._r2d2_net)
        self._blur = GaussianBlur2d((5, 5), (1.2, 1.2))
        freeze_module(self._blur)
        self._max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self._score_thresh = 0.7
        self._top_k_keypoints = 512
        self._top_k_matches = 128

    def make_pyramid(self, img, scale=0.5, num_scales=2) -> Tuple[List[torch.Tensor], torch.Tensor]:
        imgs = [img]
        scales = [1]
        for i in range(1, num_scales):
            blurred = self._blur(imgs[-1])
            imgs.append(F.interpolate(blurred, scale_factor=scale))
            scales.append(scales[-1] * scale)
        scales = 1 / torch.tensor(scales, dtype=img.dtype, device=img.device)
        return imgs, scales

    def extract_descriptors(self, img: torch.Tensor, num_keypoints=None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_keypoints is None:
            num_keypoints = self._top_k_keypoints
        img_pyr, img_scales = self.make_pyramid(img)
        img_r2d2 = self._r2d2_net(imgs=img_pyr)  # type: Dict[str, List[torch.Tensor]]

        b, c, _, _ = img_r2d2["descriptors"][0].shape

        locations = [torch.zeros(2, 0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        scores = [torch.zeros(0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        descriptors = [torch.zeros(c, 0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        scales = [torch.zeros(0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        for i in range(len(img_pyr)):
            # non maxima suppresion
            maxima_mask = img_r2d2["repeatability"][i] == self._max_filter(img_r2d2["repeatability"][i])

            # take the geometric mean of the keypoint scores
            mean_scores = torch.sqrt(img_r2d2["repeatability"][i] * img_r2d2["reliability"][i])

            # get rid of poorly scoring maxima
            maxima_mask &= mean_scores >= self._score_thresh

            # gather the maxima and their scores/ descriptors/ scales
            for b_i in range(b):
                new_locs = maxima_mask[b_i, 0].nonzero().t()
                locations[b_i] = torch.cat(
                    (locations[b_i], new_locs * img_scales[i]), dim=-1
                )
                new_scores = mean_scores[b_i, 0, new_locs[0], new_locs[1]]
                scores[b_i] = torch.cat(
                    (scores[b_i], new_scores)
                )
                new_descriptors = img_r2d2["descriptors"][i][b_i, :, new_locs[0], new_locs[1]]
                descriptors[b_i] = torch.cat(
                    (descriptors[b_i], new_descriptors), dim=-1
                )
                scales[b_i] = torch.cat(
                    (scales[b_i], img_scales[i].expand(new_locs.shape[1]))
                )

        # take the top K descriptors for each image
        for b_i in range(b):
            scores[b_i], top_ind = scores[b_i].topk(num_keypoints, sorted=False)
            locations[b_i] = locations[b_i][:, top_ind]
            descriptors[b_i] = descriptors[b_i][:, top_ind]
            scales[b_i] = scales[b_i][top_ind]

        locations = torch.stack(locations)  # b x 2 x num_keypoints
        scores = torch.stack(scores)  # b x num_keypoints
        descriptors = torch.stack(descriptors)  # b x 128 x num_keypoints
        scales = torch.stack(scales)  # b x num_keypoints

        return locations, scores, descriptors, scales

    def forward(self, img1, img2):
        img1_locations, img1_scores, img1_descriptors, _ = self.extract_descriptors(img1)
        img2_locations, img2_scores, img2_descriptors, _ = self.extract_descriptors(img2)

        img1_descriptors = img1_descriptors * img1_scores.unsqueeze(1).expand_as(img1_descriptors)
        img2_descriptors = img2_descriptors * img2_scores.unsqueeze(1).expand_as(img2_descriptors)

        scores = img1_descriptors.permute(0, 2, 1).bmm(img2_descriptors)  # b x n x n

        _, top_k_match_idx = torch.topk(scores.flatten(1), self._top_k_matches)  # b x top_k_matches
        img1_idx = top_k_match_idx // scores.shape[-1]  # b x top_k_matches
        img2_idx = top_k_match_idx % scores.shape[-1]  # b x top_k_matches

        img1_idx = img1_idx.unsqueeze(1).expand(-1, img1_locations.shape[1], -1)  # b x 2 x top_k_matches
        img2_idx = img2_idx.unsqueeze(1).expand(-1, img2_locations.shape[1], -1)  # b x 2 x top_k_matches

        img1_match_locs = torch.gather(img1_locations, 2, img1_idx)
        img2_match_locs = torch.gather(img2_locations, 2, img2_idx)

        correspondences = torch.cat(
            (img1_match_locs, img2_match_locs),
            dim=1
        )
        return correspondences


class QKVAttention(torch.nn.Module):
    def __init__(self, q_in_channels, k_in_channels, rel_channels, v_in_channels, v_out_channels, temperature=None):
        super().__init__()
        self.query_conv = torch.nn.Conv1d(q_in_channels, rel_channels, kernel_size=1, bias=False)
        self.key_conv = torch.nn.Conv1d(k_in_channels, rel_channels, kernel_size=1, bias=False)
        self.value_conv = torch.nn.Conv1d(v_in_channels, v_out_channels, kernel_size=1, bias=False)
        self.temperature = temperature

        if q_in_channels == rel_channels:
            torch.nn.init.eye_(self.query_conv.weight[:, :, 0])
        if k_in_channels == rel_channels:
            torch.nn.init.eye_(self.key_conv.weight[:, :, 0])
        if v_in_channels == v_out_channels:
            torch.nn.init.eye_(self.value_conv.weight[:, :, 0])

    def forward(self, q, k, v):
        # x should have shape b x in_channels x n
        q = self.query_conv(q)  # shape: b x rel_proj_channels x n
        k = self.key_conv(k)  # shape: b x rel_proj_channels x n
        v = self.value_conv(v)  # shape: b x v_out_channels x n
        b, c, n = q.shape

        # as temperature -> 0, softmax goes to max
        temperature = self.temperature
        if temperature is None:
            # assume q and k are random vectors with components in (0, 1)
            # setting temp to sqrt(d) where d is dimension of q an k normalizes
            # the variance of the input to 1
            temperature = torch.sqrt(torch.tensor(c, device=q.device, dtype=q.dtype))

        relevancy = (q.permute(0, 2, 1).bmm(k))

        two_best, _ = torch.topk(relevancy, 2)
        two_best_acos = torch.acos(two_best)
        ratios = two_best_acos[:, :, 0] / two_best_acos[:, :, 1]  # shape: b x n

        relevancy = F.softmax(relevancy / temperature, dim=-1)  # shape: b x n x n

        y = relevancy.bmm(v.permute(0, 2, 1)).permute(0, 2, 1)  # shape: b x v_out_channels x n

        return y, ratios


class CorrespondenceEngine(torch.nn.Module):
    def __init__(self, r2d2_net: torch.nn.Module):
        super().__init__()
        self._r2d2_net = r2d2_net
        freeze_module(self._r2d2_net)
        self._blur = GaussianBlur2d((5, 5), (1.2, 1.2))
        freeze_module(self._blur)
        self._max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self._score_thresh = 0.7
        self._top_k_keypoints = 512
        self._top_k_matches = 128

        self._initial_matcher = QKVAttention(128, 128, 128, 2, 2, temperature=1 / 512)

    def make_pyramid(self, img, scale=0.5, num_scales=2) -> Tuple[List[torch.Tensor], torch.Tensor]:
        imgs = [img]
        scales = [1]
        for i in range(1, num_scales):
            blurred = self._blur(imgs[-1])
            imgs.append(F.interpolate(blurred, scale_factor=scale))
            scales.append(scales[-1] * scale)
        scales = 1 / torch.tensor(scales, dtype=img.dtype, device=img.device)
        return imgs, scales

    def extract_descriptors(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_pyr, img_scales = self.make_pyramid(img)
        img_r2d2 = self._r2d2_net(imgs=img_pyr)  # type: Dict[str, List[torch.Tensor]]

        b, c, _, _ = img_r2d2["descriptors"][0].shape

        locations = [torch.zeros(2, 0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        scores = [torch.zeros(0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        descriptors = [torch.zeros(c, 0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        scales = [torch.zeros(0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        for i in range(len(img_pyr)):
            # non maxima suppresion
            maxima_mask = img_r2d2["repeatability"][i] == self._max_filter(img_r2d2["repeatability"][i])

            # take the geometric mean of the keypoint scores
            mean_scores = torch.sqrt(img_r2d2["repeatability"][i] * img_r2d2["reliability"][i])

            # get rid of poorly scoring maxima
            maxima_mask &= mean_scores >= self._score_thresh

            # gather the maxima and their scores/ descriptors/ scales
            for b_i in range(b):
                new_locs = maxima_mask[b_i, 0].nonzero().t()
                locations[b_i] = torch.cat(
                    (locations[b_i], new_locs * img_scales[i]), dim=-1
                )
                new_scores = mean_scores[b_i, 0, new_locs[0], new_locs[1]]
                scores[b_i] = torch.cat(
                    (scores[b_i], new_scores)
                )
                new_descriptors = img_r2d2["descriptors"][i][b_i, :, new_locs[0], new_locs[1]]
                descriptors[b_i] = torch.cat(
                    (descriptors[b_i], new_descriptors), dim=-1
                )
                scales[b_i] = torch.cat(
                    (scales[b_i], img_scales[i].expand(new_locs.shape[1]))
                )

        # take the top K descriptors for each image
        for b_i in range(b):
            scores[b_i], top_ind = scores[b_i].topk(self._top_k_keypoints, sorted=False)
            locations[b_i] = locations[b_i][:, top_ind]
            descriptors[b_i] = descriptors[b_i][:, top_ind]
            scales[b_i] = scales[b_i][top_ind]

        locations = torch.stack(locations)  # b x 2 x self._top_k_keypoints
        scores = torch.stack(scores)  # b x self._top_k_keypoints
        descriptors = torch.stack(descriptors)  # b x 128 x self._top_k_keypoints
        scales = torch.stack(scales)  # b x self._top_k_keypoints

        return locations, scores, descriptors, scales

    def forward(self, img1, img2):
        img1_locations, img1_scores, img1_descriptors, _ = self.extract_descriptors(img1)
        img2_locations, img2_scores, img2_descriptors, _ = self.extract_descriptors(img2)

        img1_descriptors = img1_descriptors * img1_scores.unsqueeze(1).expand_as(img1_descriptors)
        img2_descriptors = img2_descriptors * img2_scores.unsqueeze(1).expand_as(img2_descriptors)

        img1_corr_locs, img1_corr_ratios = self._initial_matcher(img1_descriptors, img2_descriptors, img2_locations)
        # img2_corr_locs, img2_corr_ratios = self._initial_matcher(img2_descriptors, img1_descriptors, img1_locations)

        img1_matches = torch.cat((img1_locations, img1_corr_locs), dim=1)  # shape: b x 4 x n
        # img2_matches = torch.cat((img2_locations, img2_corr_locs), dim=1)

        _, img1_top_k_corr_ind = torch.topk(img1_corr_ratios, self._top_k_matches, largest=False)
        img1_top_k_corr_ind = img1_top_k_corr_ind.unsqueeze_(1).expand(-1, img1_matches.shape[1], -1)
        # _, img2_top_k_corr_ind = torch.topk(img1_corr_ratios, self._top_k_matches, largest=False)
        # img2_top_k_corr_ind = img2_top_k_corr_ind.unsqueeze_(1).expand(-1, img2_matches.shape[1], -1)

        img1_matches = torch.gather(img1_matches, 2, img1_top_k_corr_ind)
        # img2_matches = torch.gather(img2_matches, 2, img2_top_k_corr_ind)
        return img1_matches  # , img2_matches


"""
class SelfAttention1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.key_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.query_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.value_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        # x should have shape b x in_channels x n
        q = self.query_conv(x)  # shape: b x out_channels x n
        k = self.key_conv(x)  # shape: b x out_channels x n
        v = self.value_conv(x)  # shape: b x out_channels x n
        b, c, n = q.shape
        scale_factor = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))

        relevancy = (q.permute(0, 2, 1).bmm(k)) / scale_factor
        relevancy = F.softmax(relevancy, dim=-1)  # shape: b x n x n

        y = relevancy.bmm(v.permute(0, 2, 1)).permute(0, 2, 1)  # shape: b x out_channels x n
        return y


class DualAttention1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.key_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.query_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.value_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x1, x2):
        # x1, x2 should have shape b x in_channels x n
        q1 = self.query_conv(x1)
        k1 = self.key_conv(x2)
        v1 = self.value_conv(x2)

        q2 = self.query_conv(x2)
        k2 = self.key_conv(x1)
        v2 = self.value_conv(x1)

        b, c, n = q1.shape
        scale_factor = torch.sqrt(torch.tensor(c, device=x1.device, dtype=x1.dtype))

        relevancy1 = (q1.permute(0, 2, 1).bmm(k1)) / scale_factor
        relevancy1 = F.softmax(relevancy1, dim=-1)

        relevancy2 = (q2.permute(0, 2, 1).bmm(k2)) / scale_factor
        relevancy2 = F.softmax(relevancy2, dim=-1)

        y1 = (relevancy1.bmm(v1.permute(0, 2, 1))).permute(0, 2, 1)  # shape: b x out_channels x n
        y2 = (relevancy2.bmm(v2.permute(0, 2, 1))).permute(0, 2, 1)  # shape: b x out_channels x n

        # return y2 in place of x1 since y2 contains values from x1
        return y2, y1


class DualResidual(torch.nn.Module):
    def __init__(self, branch: torch.nn.Module, iterations=1, activation=True):
        super().__init__()
        self._branch = branch
        self._iterations = iterations
        self._activation = activation

    def forward(self, x, y):
        for i in range(self._iterations):
            dx, dy = self._branch(x, y)
            x += dx
            y += dy
            if self._activation:
                x = F.leaky_relu(x, inplace=True)
                y = F.leaky_relu(y, inplace=True)
        return x, y


class Residual(torch.nn.Module):
    def __init__(self, branch: torch.nn.Module, iterations=1, chan_adapter=None, activation=True):
        super().__init__()
        self._branch = branch
        self._iterations = iterations
        self._chan_adapter = chan_adapter
        self._activation = activation

    def forward(self, x):
        for i in range(self._iterations):
            dx = self._branch(x)
            if self._chan_adapter is not None:
                x = self._chan_adapter(x)
            x += dx
            if self._activation:
                x = F.leaky_relu(x, inplace=True)
        return x


class Siamese(torch.nn.Module):
    def __init__(self, branch: torch.nn.Module, iterations=1):
        super().__init__()
        self._branch = branch

    def forward(self, x, y):
        return self._branch(x), self._branch(y)
        
class CorrespondenceEngine(torch.nn.Module):

    def __init__(self, r2d2_net: torch.nn.Module):
        super().__init__()
        self._r2d2_net = r2d2_net
        freeze_module(self._r2d2_net)
        self._blur = GaussianBlur2d((5, 5), (1.2, 1.2))
        freeze_module(self._blur)
        self._max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self._score_thresh = 0.7
        self._top_k_keypoints = 512
        self._two_stream_network = torch.nn.ModuleList([
            Siamese(torch.nn.Conv1d(130, 1024, 1, bias=False)),
            DualResidual(DualAttention1D(1024, 1024)),
            Siamese(Residual(SelfAttention1D(1024, 1024), iterations=2)),
            DualResidual(DualAttention1D(1024, 1024)),
            Siamese(Residual(SelfAttention1D(1024, 1024), iterations=2)),
            DualResidual(DualAttention1D(1024, 1024)),
            Siamese(Residual(SelfAttention1D(1024, 1024), iterations=2)),
            DualResidual(DualAttention1D(1024, 1024)),
        ])

        self._one_stream_network = torch.nn.Sequential(
            Residual(SelfAttention1D(2048, 2048), iterations=2),
            Residual(SelfAttention1D(2048, 1024), chan_adapter=torch.nn.Conv1d(2048, 1024, 1, bias=False)),
            Residual(SelfAttention1D(1024, 1024), iterations=2),
            Residual(SelfAttention1D(1024, 512), chan_adapter=torch.nn.Conv1d(1024, 512, 1, bias=False)),
            Residual(SelfAttention1D(512, 512), iterations=2),
            Residual(SelfAttention1D(512, 256), chan_adapter=torch.nn.Conv1d(512, 256, 1, bias=False)),
            Residual(SelfAttention1D(256, 256), iterations=2),
            Residual(SelfAttention1D(256, 64), chan_adapter=torch.nn.Conv1d(256, 64, 1, bias=False)),
            Residual(SelfAttention1D(64, 64), iterations=2),
            Residual(SelfAttention1D(64, 16), chan_adapter=torch.nn.Conv1d(64, 16, 1, bias=False)),
            Residual(SelfAttention1D(16, 16), iterations=2),
            Residual(SelfAttention1D(16, 4), chan_adapter=torch.nn.Conv1d(16, 4, 1, bias=False)),
        )

    def make_pyramid(self, img, scale=0.5, num_scales=2) -> Tuple[List[torch.Tensor], torch.Tensor]:
        imgs = [img]
        scales = [1]
        for i in range(1, num_scales):
            blurred = self._blur(imgs[-1])
            imgs.append(F.interpolate(blurred, scale_factor=scale))
            scales.append(scales[-1] * scale)
        scales = 1 / torch.tensor(scales, dtype=img.dtype, device=img.device)
        return imgs, scales

    def extract_descriptors(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_pyr, img_scales = self.make_pyramid(img)
        img_r2d2 = self._r2d2_net(imgs=img_pyr)  # type: Dict[str, List[torch.Tensor]]

        b, c, _, _ = img_r2d2["descriptors"][0].shape

        locations = [torch.zeros(2, 0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        scores = [torch.zeros(0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        descriptors = [torch.zeros(c, 0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        scales = [torch.zeros(0, device=img.device, dtype=img.dtype) for b_i in range(b)]
        for i in range(len(img_pyr)):
            # non maxima suppresion
            maxima_mask = img_r2d2["repeatability"][i] == self._max_filter(img_r2d2["repeatability"][i])

            # take the geometric mean of the keypoint scores
            mean_scores = torch.sqrt(img_r2d2["repeatability"][i] * img_r2d2["reliability"][i])

            # get rid of poorly scoring maxima
            maxima_mask &= mean_scores >= self._score_thresh

            # gather the maxima and their scores/ descriptors/ scales
            for b_i in range(b):
                new_locs = maxima_mask[b_i, 0].nonzero().t()
                locations[b_i] = torch.cat(
                    (locations[b_i], new_locs * img_scales[i]), dim=-1
                )
                new_scores = mean_scores[b_i, 0, new_locs[0], new_locs[1]]
                scores[b_i] = torch.cat(
                    (scores[b_i], new_scores)
                )
                new_descriptors = img_r2d2["descriptors"][i][b_i, :, new_locs[0], new_locs[1]]
                descriptors[b_i] = torch.cat(
                    (descriptors[b_i], new_descriptors), dim=-1
                )
                scales[b_i] = torch.cat(
                    (scales[b_i], img_scales[i].expand(new_locs.shape[1]))
                )

        # take the top K descriptors for each image
        for b_i in range(b):
            scores[b_i], top_ind = scores[b_i].topk(self._top_k_keypoints, sorted=False)
            locations[b_i] = locations[b_i][:, top_ind]
            descriptors[b_i] = descriptors[b_i][:, top_ind]
            scales[b_i] = scales[b_i][top_ind]

        locations = torch.stack(locations)  # b x 2 x self._top_k_keypoints
        scores = torch.stack(scores)  # b x self._top_k_keypoints
        descriptors = torch.stack(descriptors)  # b x 128 x self._top_k_keypoints
        scales = torch.stack(scales)  # b x self._top_k_keypoints

        return locations, scores, descriptors, scales

    @staticmethod
    def fuse_descriptor_data(image_shape, locations, scores, descriptors):
        descriptors = descriptors * scores.unsqueeze(1).expand_as(descriptors)
        descriptors = torch.rand_like(descriptors)  # TODO: REMOVE

        locations[:, 0] = ((locations[:, 0] / image_shape[2]) - .5) / 1000
        locations[:, 1] = ((locations[:, 1] / image_shape[3]) - .5) / 1000
        descriptors = torch.cat(
            (descriptors, locations), dim=1
        )
        return descriptors

    def forward(self, img1, img2):
        img1_locations, img1_scores, img1_descriptors, _ = self.extract_descriptors(img1)
        img2_locations, img2_scores, img2_descriptors, _ = self.extract_descriptors(img2)

        x1 = self.fuse_descriptor_data(img1.shape, img1_locations, img1_scores, img1_descriptors)
        x2 = self.fuse_descriptor_data(img2.shape, img2_locations, img2_scores, img2_descriptors)

        for module in self._two_stream_network:
            x1, x2 = module(x1, x2)

        x = torch.cat((x1, x2), dim=1)

        for module in self._one_stream_network:
            x = module(x)

        return x
"""
