import os
import random
import time
from typing import Callable, Union, Iterable, Tuple, Iterator, Dict

import numpy as np
import torch
import torch.utils.data
import torch.utils.tensorboard
from torch.optim.optimizer import Optimizer as TorchOptimizer

from epipointnet.data.pairs import CorrespondenceFundamentalMatrixPair
from epipointnet.datasets.image import load_image_for_torch
from epipointnet.datasets.shuffle import ShuffledDataset
from epipointnet.dfe.loss import robust_symmetric_epipolar_distance, symmetric_epipolar_distance
from epipointnet.imips_pytorch.trainer import ImipTrainer
from epipointnet.model import PatchBatchEpiPointNet

ImipLossType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
EpiLossType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class EpiPointNetTrainer:

    def __init__(self,
                 epi_point_net: PatchBatchEpiPointNet,
                 optimizer_factory: Callable[[Union[Iterable[torch.Tensor], dict]], TorchOptimizer],
                 train_dataset: torch.utils.data.Dataset,
                 eval_dataset: torch.utils.data.Dataset,
                 num_eval_samples: int,
                 save_directory: str,
                 inlier_radius: int = 3,
                 seed: int = 0,
                 device: str = "cuda",
                 ):
        self._seed = seed
        np.random.seed(self._seed)
        torch.random.manual_seed(self._seed)
        random.seed(self._seed)

        self._epi_point_net = epi_point_net
        self._optimizer = optimizer_factory(self._epi_point_net.parameters())
        self._train_dataset = train_dataset
        self._train_dataset_iter = self._create_new_train_dataset_iterator()
        self._train_eval_dataset = ShuffledDataset(train_dataset, num_eval_samples)
        self._eval_dataset = ShuffledDataset(eval_dataset, num_eval_samples)
        self._inlier_radius = inlier_radius
        self._device = device
        self._imip_eps = torch.tensor(1e-4, device=self._device)
        self._best_eval_score = torch.tensor([0], device=self._device)
        self._t_board_writer = torch.utils.tensorboard.SummaryWriter()
        self._best_checkpoint_path = os.path.join(save_directory, "imips_best.chkpt")
        self._latest_checkpoint_path = os.path.join(save_directory, "imips_latest.chkpt")

    # ################ DATASET ITERATION ################

    def _create_new_train_dataset_iterator(self) -> Iterator[CorrespondenceFundamentalMatrixPair]:
        """
        Shuffles self._train_dataset and creates a new torch dataloader backed iterator
        :return: A iterator which loads data from the training dataset one item at a time
        """
        shuffled_dataset = ShuffledDataset(self._train_dataset)
        return iter(torch.utils.data.DataLoader(shuffled_dataset, batch_size=None, collate_fn=lambda x: x))

    def _next_pair(self) -> CorrespondenceFundamentalMatrixPair:
        """
        Get the next stereo pair to train on. Reshuffles the training dataset
        and re-starts the iterator if the iterator runs out of data.
        :return: A stereo pair to train on with fundamental matrix data and correspondence data
        """
        try:
            return next(self._train_dataset_iter)
        except StopIteration:
            self._train_dataset_iter = self._create_new_train_dataset_iterator()
            return next(self._train_dataset_iter)

    def _train_patches(self, img_1_anchor_patches: torch.Tensor, img_2_anchor_patches: torch.Tensor,
                       img_1_keypoints_xy: torch.Tensor, img_2_keypoints_xy: torch.Tensor,
                       img_1_inlier_labels: torch.Tensor, img_1_outlier_labels: torch.Tensor,
                       img_1_corr_patches: torch.Tensor,
                       image_1_epi_points: torch.Tensor, image_2_epi_points: torch.Tensor,
                       ) -> Dict[str, torch.Tensor]:
        """
        Runs the PatchBatchEpiPointNet model against the input patches, derives the losses,
        and runs the model optimizer.

        :param img_1_anchor_patches: Patches about the keypoints detected by the IMIP network in the first image
        :param img_2_anchor_patches: Patches about the keypoints detected by the IMIP network in the second image
        :param img_1_keypoints_xy: Keypoints detected by the IMIP network in the first image
        :param img_2_keypoints_xy: Keypoints detected by the IMIP network in the second image
        :param img_1_inlier_labels: 1D bool tensor which is true when the keypoints detected by the IMIP network in the
         first image are near the true correspondences of the keypoints detected by the IMIP network in the second image
        :param img_1_outlier_labels: 1D bool tensor which is true when the keypoints detected by the IMIP network in the
         first image aren't near the true correspondences of the keypoints detected by the IMIP network in the second
          image
        :param img_1_corr_patches: Patches about the true correspondences of the keypoints detected by the IMIP network
         in the second image
        :param image_1_epi_points: Points in image 1 which perfectly align with the ground truth fundamental matrix
         and image_2_epi_points
        :param image_2_epi_points: Points in image 2 which perfectly align with the ground truth fundamental matrix
         and image_1_epi_points
        :return: A dictionary of the various losses
        """

        # Feed the PatchBatchEpiPointNet the local patches about the keypoints detected by the IMIP network
        # in both images. We also feed the network the locations of the keypoints so it can
        # correctly derive the resulting f_mats. In testing, we feed the network whole images
        # and the argmax operation extracts the keypoint locations for us.
        (f_mats, img_1_keypoints_xy_net_output, img_2_keypoints_xy_net_output, corr_weights,
         img_1_imip_outs, img_2_imip_outs) = self._epi_point_net(
            img_1_anchor_patches, img_2_anchor_patches,
            img_1_keypoints_xy, img_2_keypoints_xy
        )

        # Feed the correspondence patches through the IMIP network in order to compute the IMIP loss
        corr_img_1_imip_outs = self._epi_point_net.imip_net(img_1_corr_patches, keepDim=False)

        (imip_loss, outlier_correspondence_loss, inlier_loss,
         outlier_maximizer_loss, unaligned_maximizer_loss) = ImipTrainer.loss(
            img_1_imip_outs, corr_img_1_imip_outs,
            img_1_inlier_labels, img_1_outlier_labels, self._imip_eps
        )

        # TODO: experiment with setting epi loss to zero if < 8 inliers
        # Find the distance between the points which fit the ground truth fundamental matrix
        # and the epipolar lines generated by the fundamental matrices produced by the DFE network
        # Note that we cap the point-wise losses in order to promote training stability.
        epi_loss = 0
        for f_mat in f_mats:
            epi_loss += robust_symmetric_epipolar_distance(image_1_epi_points, image_2_epi_points, f_mat)
        epi_loss = epi_loss.mean()

        # TODO: incorporate epi loss once we can successfully minimize imip_loss
        loss = imip_loss  # + epi_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        f_estimated = f_mats[-1]
        f_estimated = f_estimated / f_estimated[:, -1, -1]
        f_estimated_loss = symmetric_epipolar_distance(image_1_epi_points, image_2_epi_points, f_estimated).mean()
        return {
            "loss": loss.detach(),
            "epi_loss": epi_loss.detach(),
            "f_estimated_loss": f_estimated_loss.detach(),
            "imip_loss": imip_loss.detach(),
            "outlier_correspondence_loss": outlier_correspondence_loss.detach(),
            "inlier_maximizer_loss": inlier_loss.detach(),
            "outlier_maximizer_loss": outlier_maximizer_loss.detach(),
            "unaligned_maximizer_loss": unaligned_maximizer_loss.detach(),
        }

    def _train_pair(self, pair: CorrespondenceFundamentalMatrixPair, iteration: int) -> [torch.Tensor]:
        """
        Trains the PatchBatchEpiPointNet on a stereo pair with both correspondence and fundamental matrix information
        :param pair: A stereo pair with both correspondence and fundamental matrix information
        :param iteration: how many pairs have been processed so far after processing this pair
        :return: the losses generated by processing the pair with image 2 as the origin image
        """
        # Convert numpy HW or HWC to torch CHW
        img_1 = load_image_for_torch(pair.image_1, self._device)
        img_2 = load_image_for_torch(pair.image_2, self._device)

        # Extract the anchor keypoints in each image with the IMIP network
        exclude_border_px = (self._epi_point_net.patch_size() - 1) // 2
        img_1_keypoints_xy, img_1_responses = self._epi_point_net.imip_net.extract_keypoints(img_1, exclude_border_px)
        img_2_keypoints_xy, img_2_responses = self._epi_point_net.imip_net.extract_keypoints(img_2, exclude_border_px)

        # Find the true correspondences for each anchor keypoint
        # img_1_correspondences_xy are the correspondences in img_2 of the anchor keypoints
        #   in img_1 (img_1_keypoints_xy)
        # img_2_correspondences_xy are the correspondences in img_1 of the anchor keypoints
        #   in img_2 (img_2_keypoints_xy)
        (img_1_correspondences_xy, img_1_correspondences_mask,
         img_2_correspondences_xy, img_2_correspondences_mask) = ImipTrainer.find_correspondences(
            pair, img_1_keypoints_xy, img_2_keypoints_xy, exclude_border_px
        )

        # Create the inlier/ outlier labels
        # img_1_inlier_labels is true at an index when there is a valid, true correspondence in img_1 for the
        #   index aligned anchor in img_2 and the anchor in img_1 is close to the true correspondence in img_1
        # img_2_inlier_labels is true at an index when there is a valid, true correspondence in img_2 for the
        #   index aligned anchor in img_1 and the anchor in img_2 is close to the true correspondence in img_2
        (img_1_inlier_labels, img_1_outlier_labels,
         img_2_inlier_labels, img_2_outlier_labels) = ImipTrainer.label_inliers_outliers(
            img_1_keypoints_xy, img_1_correspondences_xy, img_1_correspondences_mask,
            img_2_keypoints_xy, img_2_correspondences_xy, img_2_correspondences_mask,
            self._inlier_radius
        )

        apparent_inliers = (img_1_inlier_labels & img_2_inlier_labels).sum()
        apparent_outliers = (img_1_outlier_labels | img_2_outlier_labels).sum()
        self._t_board_writer.add_scalar("training/apparent inliers", apparent_inliers, iteration)
        self._t_board_writer.add_scalar("training/apparent outliers", apparent_outliers, iteration)

        # Grab the neighborhoods about the keypoints and their true correspondences
        # Note the meaning of the img_1 and img_2 prefixes changes here.
        # img_X means the patch comes from img_X. For example, img_1_corr_patches are patches sampled from image 1
        # surrounding img_2_correspondences_xy.
        (img_1_anchor_patches, img_1_corr_patches,
         img_2_anchor_patches, img_2_corr_patches) = ImipTrainer.generate_training_patches(
            pair,
            img_1_keypoints_xy, img_1_correspondences_xy, img_1_correspondences_mask,
            img_2_keypoints_xy, img_2_correspondences_xy, img_2_correspondences_mask,
            self._epi_point_net.patch_size(),
        )

        # Generate points which fit the ground truth fundamental matrix
        pts_1_virt, pts_2_virt = pair.generate_virtual_points()
        pts_1_virt = torch.tensor(pts_1_virt, device=self._device, dtype=torch.float32, requires_grad=False).unsqueeze(
            0)
        pts_2_virt = torch.tensor(pts_2_virt, device=self._device, dtype=torch.float32, requires_grad=False).unsqueeze(
            0)

        # Train with image 1 as the origin image
        self._train_patches(
            img_1_anchor_patches, img_2_anchor_patches,
            img_1_keypoints_xy, img_2_keypoints_xy,
            img_1_inlier_labels, img_1_outlier_labels, img_1_corr_patches,
            pts_1_virt, pts_2_virt
        )

        # Train with image 2 as the origin image
        losses = self._train_patches(
            img_2_anchor_patches, img_1_anchor_patches,
            img_2_keypoints_xy, img_1_keypoints_xy,
            img_2_inlier_labels, img_2_outlier_labels, img_2_corr_patches,
            pts_2_virt, pts_1_virt
        )

        self._t_board_writer.add_scalar("training/loss", losses["loss"], iteration)

        self._t_board_writer.add_scalar("training/epi_loss", losses["epi_loss"], iteration)
        self._t_board_writer.add_scalar("training/f_est_loss", losses["f_estimated_loss"], iteration)

        self._t_board_writer.add_scalar("training/imip_loss", losses["imip_loss"], iteration)
        self._t_board_writer.add_scalar("training/Outlier Correspondence Loss", losses["outlier_correspondence_loss"],
                                        iteration)
        self._t_board_writer.add_scalar("training/Outlier Maximizer Loss", losses["outlier_maximizer_loss"], iteration)
        self._t_board_writer.add_scalar("training/Inlier Maximizer Loss", losses["inlier_maximizer_loss"], iteration)
        self._t_board_writer.add_scalar("training/Unaligned Maximizer Loss", losses["unaligned_maximizer_loss"],
                                        iteration)

        return losses["loss"]

    def train(self, iterations: int, eval_frequency: int):
        if iterations % eval_frequency != 0:
            raise ValueError(
                "eval_frequency ({}) must evenly divide iterations ({})".format(eval_frequency, iterations))

        last_eval_time = time.time()
        for iteration in range(1, iterations + 1):
            curr_pair = self._next_pair()
            curr_loss = self._train_pair(curr_pair, iteration)

            if iteration % eval_frequency == 0:
                save_info = {
                    "inlier_radius": self._inlier_radius,
                    "iteration": iteration,
                    "network_state_dict": self._epi_point_net.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "training/loss": curr_loss,
                    "seed": self._seed,
                }

                torch.save(save_info, self._latest_checkpoint_path)

                # TODO: Write _evaluate_dataset
                # if eval_true_inliers_avg > self._best_eval_inlier_score:
                #     self._best_eval_inlier_score = eval_true_inliers_avg
                #     torch.save(save_info, self._best_checkpoint_path)

                curr_eval_time = time.time()
                iters_per_minute = eval_frequency / ((curr_eval_time - last_eval_time) / 60)
                self._t_board_writer.add_scalar("performance/iterations per minute", iters_per_minute, iteration)
                last_eval_time = curr_eval_time
