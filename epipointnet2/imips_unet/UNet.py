import multiprocessing
from argparse import ArgumentParser
from typing import Tuple, List, Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader, Subset

from epipointnet.data.pairs import CorrespondencePair
from epipointnet.datasets.kitti import KITTIMonocularStereoPairs
from epipointnet.datasets.tum_mono import TUMMonocularStereoPairs


class UNet(pl.LightningModule):
    def __init__(self, hparams):
        super(UNet, self).__init__()
        self.save_hyperparameters()

        """ for loading checkpoints without hyperparameters
        class temp_hparams:
            train_set = "kitti"
            eval_set = "kitti"
            test_set = "kitti"
            data_root = "./data"
            n_channels = 3
            n_classes = 128
            batch_size = 4
            n_eval_samples = 64
            max_inlier_distance = 6
        hparams = temp_hparams()
        """

        self.data_root = hparams.data_root
        self.n_channels = hparams.n_channels
        self.n_classes = hparams.n_classes
        self.batch_size = hparams.batch_size

        kitti_scale = 0.33  # 122, 404
        tum_scale = 0.25

        if hparams.train_set == "kitti":
            self.train_set = KITTIMonocularStereoPairs(self.data_root, "train", color=True)
            self.train_scale = kitti_scale
        elif hparams.train_set == "tum":
            self.train_set = TUMMonocularStereoPairs(self.data_root, "train")
            self.train_scale = tum_scale
            assert self.n_channels == 1

        if hparams.eval_set == "kitti":
            self.eval_set = Subset(KITTIMonocularStereoPairs(self.data_root, "validation", color=True),
                                   range(hparams.n_eval_samples))
            self.eval_scale = kitti_scale
        elif hparams.eval_set == "tum":
            self.eval_set = Subset(TUMMonocularStereoPairs(self.data_root, "validation"),
                                   range(hparams.n_eval_samples))
            self.eval_scale = tum_scale
            assert self.n_channels == 1

        if hparams.test_set == "kitti":
            self.test_set = Subset(KITTIMonocularStereoPairs(self.data_root, "test", color=True),
                                   range(hparams.n_eval_samples))
            self.test_scale = kitti_scale
        elif hparams.test_set == "tum":
            self.test_set = Subset(TUMMonocularStereoPairs(self.data_root, "test"),
                                   range(hparams.n_eval_samples))
            self.test_scale = tum_scale
            assert self.n_channels == 1

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        # noinspection PyPep8Naming
        class up(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diff_y = x2.size()[2] - x1.size()[2]
                diff_x = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [
                        diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2
                    ], mode='reflect'
                )
                x_cat = torch.cat([x2, x1], dim=1)
                return self.conv(x_cat)

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True),
        )
        # generate gaussian
        self.register_buffer(
            "max_inlier_distance",
            torch.tensor(hparams.max_inlier_distance, device=self.device)
        )  # In scaled space
        sigma = hparams.max_inlier_distance / 3
        size = int(6 * sigma + 1)
        x = torch.arange(0, size, dtype=self.dtype, device=self.device)
        y = x.clone().unsqueeze(-1)
        x0 = y0 = size // 2  # center of the gaussian / cauchy
        label_mask = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        label_mask_thresh = 0.1
        binary_label_mask = label_mask > label_mask_thresh
        self.register_buffer("label_mask", label_mask)
        self.register_buffer("binary_label_mask", binary_label_mask)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        _, max_idx = torch.max(x.detach().view(x.shape[0], x.shape[1], -1), 2)
        coords = max_idx.unsqueeze(-1).repeat(1, 1, 2).to(dtype=self.dtype)
        coords[:, :, 0] = coords[:, :, 0] % x.shape[3]
        coords[:, :, 1] = torch.floor(coords[:, :, 1] / x.shape[3])
        coords = coords.permute(0, 2, 1)

        return x, coords

    def derive_loss(self, image_1_logits: torch.Tensor, image_1_keypoints_xy: torch.Tensor,
                    image_2_logits: torch.Tensor, image_2_keypoints_xy: torch.Tensor,
                    correspondence_funcs: List[Callable[..., Tuple[np.ndarray, np.ndarray]]],
                    scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # find the true correspondences to the predicted keypoints
        image_1_corrs_xy = torch.zeros_like(image_1_keypoints_xy)
        image_1_corrs_mask = torch.zeros(image_1_corrs_xy.shape[0], image_1_corrs_xy.shape[2], dtype=torch.bool,
                                         device=self.device)
        image_2_corrs_xy = torch.zeros_like(image_2_keypoints_xy)
        image_2_corrs_mask = torch.zeros(image_2_corrs_xy.shape[0], image_2_corrs_xy.shape[2], dtype=torch.bool,
                                         device=self.device)

        # Get the correspondences in the original image scale
        rescaled_image_1_keypoints_xy = image_1_keypoints_xy / scale
        rescaled_image_2_keypoints_xy = image_2_keypoints_xy / scale

        for i in range(len(correspondence_funcs)):
            corr_func = correspondence_funcs[i]
            image_1_packed_corrs_xy, image_1_corrs_indices = corr_func(
                rescaled_image_1_keypoints_xy[i, :, :].cpu().numpy(), inverse=False)

            image_1_packed_corrs_xy = torch.tensor(image_1_packed_corrs_xy, device=self.device)
            image_1_corrs_indices = torch.tensor(image_1_corrs_indices, device=self.device)
            image_1_corrs_xy[i, :, image_1_corrs_indices] = image_1_packed_corrs_xy
            image_1_corrs_mask[i, image_1_corrs_indices] = True

            image_2_packed_corrs_xy, image_2_corrs_indices = corr_func(
                rescaled_image_2_keypoints_xy[i, :, :].cpu().numpy(), inverse=True)
            image_2_packed_corrs_xy = torch.tensor(image_2_packed_corrs_xy, device=self.device)
            image_2_corrs_indices = torch.tensor(image_2_corrs_indices, device=self.device)
            image_2_corrs_xy[i, :, image_2_corrs_indices] = image_2_packed_corrs_xy
            image_2_corrs_mask[i, image_2_corrs_indices] = True

        # Scale the correspondences back down
        del rescaled_image_1_keypoints_xy
        del rescaled_image_2_keypoints_xy
        image_1_corrs_xy *= scale
        image_2_corrs_xy *= scale

        # Figure out whether the predicted keypoints in one image are near the
        # the true correspondences of the predicted keypoints in the other image

        # image_1_inlier_distances measures how far off the keypoints predicted in image 1
        # are from the true correspondences of the keypoints predicted from image 2
        image_1_inlier_distances = torch.norm((image_1_keypoints_xy - image_2_corrs_xy), p=2,
                                              dim=1)  # type: torch.Tensor
        image_1_inlier_distance_mask = image_1_inlier_distances < self.max_inlier_distance

        # inliers and outliers in this anchor image are determined by the true correspondences of the keypoints
        # detected when its paired image is ran through the net as an anchor image
        image_1_inliers = image_1_inlier_distance_mask & image_2_corrs_mask
        image_1_outliers = ~image_1_inlier_distance_mask & image_2_corrs_mask

        image_2_inlier_distances = torch.norm((image_2_keypoints_xy - image_1_corrs_xy), p=2,
                                              dim=1)  # type: torch.Tensor
        image_2_inlier_distance_mask = image_2_inlier_distances < self.max_inlier_distance

        image_2_inliers = image_2_inlier_distance_mask & image_1_corrs_mask
        image_2_outliers = ~image_2_inlier_distance_mask & image_1_corrs_mask

        # generate label heatmap and weights for inliers, outlier masks, and decorrelation masks
        image_1_label_map, image_1_lm_weights, image_1_outlier_mask, image_1_decorr_mask = self.draw_label_maps(
            image_1_logits, image_1_keypoints_xy, image_1_inliers, image_1_outliers)
        image_2_label_map, image_2_lm_weights, image_2_outlier_mask, image_2_decorr_mask = self.draw_label_maps(
            image_2_logits, image_2_keypoints_xy, image_2_inliers, image_2_outliers)

        # compare predicted heatmaps against generated heatmaps for inlier anchors
        image_1_anchor_loss = (image_1_lm_weights * torch.nn.functional.binary_cross_entropy_with_logits(
            image_1_logits, image_1_label_map, reduction="none"
        )).sum() / (image_1_lm_weights > 0).sum().clamp_min(1)

        image_2_anchor_loss = (image_2_lm_weights * torch.nn.functional.binary_cross_entropy_with_logits(
            image_2_logits, image_2_label_map, reduction="none"
        )).sum() / (image_2_lm_weights > 0).sum().clamp_min(1)

        # push down outlier responses
        image_1_outlier_logits = image_1_logits[image_1_outlier_mask]
        if image_1_outlier_logits.numel() > 0:
            image_1_outlier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                image_1_outlier_logits, torch.zeros_like(image_1_outlier_logits), reduction="mean"
            )
        else:
            image_1_outlier_loss = torch.tensor(0, dtype=self.dtype, device=self.device)

        image_2_outlier_logits = image_2_logits[image_2_outlier_mask]
        if image_2_outlier_logits.numel() > 0:
            image_2_outlier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                image_2_outlier_logits, torch.zeros_like(image_2_outlier_logits), reduction="mean"
            )
        else:
            image_2_outlier_loss = torch.tensor(0, dtype=self.dtype, device=self.device)

        # push down other channels' responses to the input which produced an inlier for a given channel
        image_1_decorr_logits = image_1_logits[image_1_decorr_mask]
        if image_1_decorr_logits.numel() > 0:
            image_1_decorr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                image_1_decorr_logits, torch.zeros_like(image_1_decorr_logits), reduction="mean"
            )
        else:
            image_1_decorr_loss = torch.tensor(0, dtype=self.dtype, device=self.device)

        image_2_decorr_logits = image_2_logits[image_2_decorr_mask]
        if image_2_decorr_logits.numel() > 0:
            image_2_decorr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                image_2_decorr_logits, torch.zeros_like(image_2_decorr_logits), reduction="mean"
            )
        else:
            image_2_decorr_loss = torch.tensor(0, dtype=self.dtype, device=self.device)

        # aggregate the losses
        image_1_loss = image_1_anchor_loss + image_1_outlier_loss + 50 * image_1_decorr_loss
        image_2_loss = image_2_anchor_loss + image_2_outlier_loss + 50 * image_2_decorr_loss
        loss = image_1_loss + image_2_loss

        total_apparent_inliers, total_true_inliers = self.count_inliers(
            image_1_inliers & image_2_inliers, image_1_keypoints_xy
        )
        total_outliers = (image_1_outliers | image_2_outliers).sum(dim=1).to(self.dtype)
        assert torch.all(total_true_inliers <= total_apparent_inliers)  # TODO: remove
        return loss, total_apparent_inliers, total_true_inliers, total_outliers

    def draw_label_maps(self, logits: torch.Tensor, keypoints_xy: torch.Tensor,
                        inliers: torch.Tensor, outliers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                torch.Tensor, torch.Tensor]:
        image_br = torch.tensor(logits.shape[2:4],
                                device=self.device,
                                dtype=torch.long).flip(0).view(1, 2, 1).expand_as(keypoints_xy)
        px_per_channel = torch.tensor(logits.shape[2] * logits.shape[3], dtype=self.dtype, device=self.device)

        anchor_label_map = torch.zeros_like(logits)
        anchor_weights = torch.zeros_like(logits)
        outlier_mask = torch.zeros_like(logits, dtype=torch.bool)
        decorr_mask = torch.zeros_like(logits, dtype=torch.bool)

        lm_ul = (torch.round(keypoints_xy) - self.label_mask.shape[0] // 2).to(dtype=torch.long)
        lm_br = (torch.round(keypoints_xy) + self.label_mask.shape[0] // 2 + 1).to(dtype=torch.long)

        mask_from_ul = (-lm_ul).clamp_min_(0)
        mask_from_br = torch.min(lm_br, image_br) - lm_ul

        mask_to_ul = lm_ul.clamp_min_(0)
        mask_to_br = torch.min(lm_br, image_br)

        # Draw in the negative targets to prevent channels from learning the same solutions
        # Only one channel will have positive weights for any given keypoint
        for b in range(logits.shape[0]):
            for c in range(logits.shape[1]):
                if inliers[b, c]:
                    anchor_label_map[b, c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = \
                        self.label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                        mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]]

                    num_positive = self.label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                                   mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]].numel()
                    num_negative = px_per_channel - num_positive
                    anchor_weights[b, c] = 1
                    anchor_weights[b, c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = num_negative / num_positive

                    """
                    decorr_mask[b, 0:c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                                mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] |= \
                        self.binary_label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                                               mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]]
                    """
                    decorr_mask[b, c + 1:, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] |= \
                        self.binary_label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                        mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]]
                if outliers[b, c]:
                    outlier_mask[b, c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] |= \
                        self.binary_label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                        mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]]

        return anchor_label_map, anchor_weights, outlier_mask, decorr_mask

    """ Did not work, channels converged to same solution (found out i was zeroing out the training data x.x)
    def draw_label_map(self, logits: torch.Tensor, keypoints_xy: torch.Tensor,
                       inliers: torch.Tensor, outliers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_br = torch.tensor(logits.shape[2:4],
                                device=self.device,
                                dtype=torch.long).flip(0).view(1, 2, 1).expand_as(keypoints_xy)

        label_map = torch.zeros_like(logits)
        weights = torch.zeros_like(logits)
        lm_ul = (torch.round(keypoints_xy) - self.label_mask.shape[0] // 2).to(dtype=torch.long)
        lm_br = (torch.round(keypoints_xy) + self.label_mask.shape[0] // 2 + 1).to(dtype=torch.long)

        mask_from_ul = (-lm_ul).clamp_min_(0)
        mask_from_br = torch.min(lm_br, image_br) - lm_ul

        mask_to_ul = lm_ul.clamp_min_(0)
        mask_to_br = torch.min(lm_br, image_br)

        # Draw in the negative targets to prevent channels from learning the same solutions
        # Only one channel will have positive weights for any given keypoint
        for b in range(logits.shape[0]):
            for c in range(logits.shape[1]):
                if outliers[b, c]:
                    label_map[b, :, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = 0
                    weights[b, c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = 1

        for b in range(logits.shape[0]):
            for c in range(logits.shape[1]):
                if inliers[b, c]:
                    label_map[b, :c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] -= \
                        self.label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                        mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]]
                    label_map[b, :c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]].clamp_min_(0)
                    weights[b, :c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = 1

                    label_map[b, c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = \
                        self.label_mask[mask_from_ul[b, 1, c]:mask_from_br[b, 1, c],
                        mask_from_ul[b, 0, c]:mask_from_br[b, 0, c]]
                    weights[b, c, mask_to_ul[b, 1, c]:mask_to_br[b, 1, c],
                    mask_to_ul[b, 0, c]:mask_to_br[b, 0, c]] = 1

        negative_mask = (weights == 1) & (label_map == 0)
        positive_mask = (weights == 1) & (label_map != 0)
        num_negative = negative_mask.sum().to(self.dtype)
        num_positive = positive_mask.sum().to(self.dtype)
        pos_weight = num_negative / num_positive
        weights[positive_mask] = pos_weight

        return label_map, weights
    """

    def count_inliers(self, inlier_labels: torch.Tensor, keypoints_xy: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        num_true_inliers = torch.zeros(inlier_labels.shape[0], device=self.device, dtype=self.dtype)
        num_apparent_inliers = torch.sum(inlier_labels, dim=1).to(dtype=self.dtype)

        for b in range(inlier_labels.shape[0]):
            apparent_inliers_xy = keypoints_xy[b, :, inlier_labels[b]]
            if num_apparent_inliers[b] > 0:
                unique_inliers_xy = apparent_inliers_xy[:, 0:1]
                num_true_inliers[b] += 1
                for i in range(1, int(num_apparent_inliers[b])):
                    test_inlier = apparent_inliers_xy[:, i:i + 1]
                    if (torch.norm(unique_inliers_xy - test_inlier, p=2, dim=0) > self.max_inlier_distance).all():
                        unique_inliers_xy = torch.cat((unique_inliers_xy, test_inlier), dim=1)
                        num_true_inliers[b] += 1
        return num_apparent_inliers, num_true_inliers

    def training_step(self, batch, batch_nb):
        # split out the batch data
        image_1_tensors, image_2_tensors, names, correspondence_funcs = batch

        # resize the images so we have enough RAM
        image_1_tensors = F.interpolate(image_1_tensors, scale_factor=self.train_scale,
                                        mode="bilinear", align_corners=True)
        image_2_tensors = F.interpolate(image_2_tensors, scale_factor=self.train_scale,
                                        mode="bilinear", align_corners=True)

        # run the images through the unet
        image_1_logits, image_1_keypoints_xy = self(image_1_tensors)
        image_2_logits, image_2_keypoints_xy = self(image_2_tensors)

        # generate a loss
        loss, apparent_inliers, true_inliers, outliers = self.derive_loss(image_1_logits, image_1_keypoints_xy,
                                                                          image_2_logits,
                                                                          image_2_keypoints_xy, correspondence_funcs,
                                                                          self.train_scale)

        tensorboard_logs = {'train/loss': loss, "train/apparent_inliers": apparent_inliers.mean(),
                            "train/true_inliers": true_inliers.mean(), "train/outliers": outliers.mean()}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # split out the batch data
        image_1_tensors, image_2_tensors, names, correspondence_funcs = batch

        # resize the images so we have enough RAM
        image_1_tensors = F.interpolate(image_1_tensors, scale_factor=self.eval_scale,
                                        mode="bilinear", align_corners=True)
        image_2_tensors = F.interpolate(image_2_tensors, scale_factor=self.eval_scale,
                                        mode="bilinear", align_corners=True)

        # run the images through the unet
        image_1_logits, image_1_keypoints_xy = self(image_1_tensors)
        image_2_logits, image_2_keypoints_xy = self(image_2_tensors)

        # generate a loss
        loss, apparent_inliers, true_inliers, outliers = self.derive_loss(image_1_logits, image_1_keypoints_xy,
                                                                          image_2_logits, image_2_keypoints_xy,
                                                                          correspondence_funcs, self.eval_scale)
        return {'val_loss': loss, "val_apparent_inliers": apparent_inliers, "val_true_inliers": true_inliers,
                "val_outliers": outliers}

    def validation_epoch_end(self, outputs):
        return {
            'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean(),
            "val_apparent_inliers": torch.stack([x['val_apparent_inliers'] for x in outputs]).mean(),
            "val_true_inliers": torch.stack([x['val_true_inliers'] for x in outputs]).mean(),
            'log': {
                'val/loss': torch.stack([x['val_loss'] for x in outputs]).mean(),
                "val/apparent_inliers": torch.stack([x['val_apparent_inliers'] for x in outputs]).mean(),
                "val/true_inliers": torch.stack([x['val_true_inliers'] for x in outputs]).mean(),
                "val/outliers": torch.stack([x['val_outliers'] for x in outputs]).mean()
            }
        }

    def test_step(self, batch, batch_nb):
        # split out the batch data
        image_1_tensors, image_2_tensors, names, correspondence_funcs = batch

        # resize the images so we have enough RAM
        image_1_tensors = F.interpolate(image_1_tensors, scale_factor=self.test_scale,
                                        mode="bilinear", align_corners=True)
        image_2_tensors = F.interpolate(image_2_tensors, scale_factor=self.test_scale,
                                        mode="bilinear", align_corners=True)

        # run the images through the unet
        image_1_logits, image_1_keypoints_xy = self(image_1_tensors)
        image_2_logits, image_2_keypoints_xy = self(image_2_tensors)

        # generate a loss
        loss, apparent_inliers, true_inliers, outliers = self.derive_loss(image_1_logits, image_1_keypoints_xy,
                                                                          image_2_logits,
                                                                          image_2_keypoints_xy, correspondence_funcs,
                                                                          self.test_scale)
        return {'test_loss': loss, "test_apparent_inliers": apparent_inliers, "test_true_inliers": true_inliers,
                "test_outliers": outliers}

    def test_epoch_end(self, outputs):
        return {
            'test_loss': torch.stack([x['test_loss'] for x in outputs]).mean(),
            "test_apparent_inliers": torch.stack([x['test_apparent_inliers'] for x in outputs]).mean(),
            "test_true_inliers": torch.stack([x['test_true_inliers'] for x in outputs]).mean(),
            'log': {
                'test/loss': torch.stack([x['test_loss'] for x in outputs]).mean(),
                "test/apparent_inliers": torch.stack([x['test_apparent_inliers'] for x in outputs]).mean(),
                "test/true_inliers": torch.stack([x['test_true_inliers'] for x in outputs]).mean(),
                "test/outliers": torch.stack([x['test_outliers'] for x in outputs]).mean()
            }
        }

    def configure_optimizers(self):
        return RMSprop(self.parameters(), lr=0.001, weight_decay=1e-8)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          collate_fn=CorrespondencePair.collate_for_torch,
                          num_workers=1 + multiprocessing.cpu_count() // 2,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_set,
                          batch_size=self.batch_size,
                          collate_fn=CorrespondencePair.collate_for_torch,
                          num_workers=1 + multiprocessing.cpu_count() // 2)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          collate_fn=CorrespondencePair.collate_for_torch,
                          num_workers=1 + multiprocessing.cpu_count() // 2)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', default="./data")
        parser.add_argument('--train_set', choices=["kitti", "tum_mono"], default="kitti")
        parser.add_argument('--eval_set', choices=["kitti", "tum_mono"], default="kitti")
        parser.add_argument('--test_set', choices=["kitti", "tum_mono"], default="kitti")
        parser.add_argument('--n_eval_samples', type=int, default=64)
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=128)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--max_inlier_distance', type=float, default=6)
        return parser
