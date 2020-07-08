import itertools
import os
import random
import time
from typing import Callable, Union, Iterable, Iterator, Dict, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torch.utils.tensorboard
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.tensorboard import SummaryWriter

from epipointnet.data.image import load_image_for_torch
from epipointnet.data.pairs import CorrespondenceFundamentalMatrixPair
from epipointnet.datasets.shuffle import ShuffledDataset
from epipointnet.dfe.loss import robust_symmetric_epipolar_distance, symmetric_epipolar_distance
from epipointnet.imips_pytorch.losses.imips import ImipLoss
from epipointnet.imips_pytorch.trainer import ImipTrainer
from epipointnet.model import PatchBatchEpiPointNet

EpiLossType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class EpiPointNetTrainer:

    def __init__(self,
                 epi_point_net: PatchBatchEpiPointNet,
                 imip_loss: ImipLoss,
                 optimizer_factory: Callable[[Union[Iterable[torch.Tensor], dict]], TorchOptimizer],
                 train_dataset: torch.utils.data.Dataset,
                 eval_dataset: torch.utils.data.Dataset,
                 num_eval_samples: int,
                 save_directory: str,
                 device: Optional[Union[str, torch.device]] = None,
                 inlier_radius: int = 3,
                 seed: int = 0,
                 ):
        self._seed = seed
        np.random.seed(self._seed)
        torch.random.manual_seed(self._seed)
        random.seed(self._seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._epi_point_net = epi_point_net.to(device=self._device)
        self._imip_loss_module = imip_loss.to(device=self._device)
        self._imip_loss_func = imip_loss.forward_with_log_data
        self._optimizer = optimizer_factory(
            itertools.chain(self._epi_point_net.parameters(), self._imip_loss_module.parameters()))
        self._train_dataset = train_dataset
        self._train_dataset_iter = self._create_new_train_dataset_iterator()
        self._train_eval_dataset = ShuffledDataset(train_dataset, num_eval_samples)
        self._eval_dataset = ShuffledDataset(eval_dataset, num_eval_samples)
        self._inlier_radius = inlier_radius
        self._best_eval_f_err_score = torch.tensor([0], device=self._device)
        self._best_checkpoint_path = os.path.join(save_directory, "epipointnet_best.chkpt")
        self._latest_checkpoint_path = os.path.join(save_directory, "epipointnet_latest.chkpt")

    # ################ CHECKPOINTING ################

    def _load_checkpoint(self, checkpoint: dict) -> int:
        self._seed = checkpoint["seed"]
        np.random.seed(self._seed)
        torch.random.manual_seed(self._seed)
        random.seed(self._seed)

        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epi_point_net.load_state_dict(checkpoint["network_state_dict"])
        self._inlier_radius = checkpoint["inlier_radius"]
        self._create_new_train_dataset_iterator()
        return checkpoint["iteration"]
        # Resetting the data iterator takes too long
        # for i in range(0, checkpoint["iteration"]):
        #     self._next_pair()

    def load_best_checkpoint(self) -> int:
        return self._load_checkpoint(torch.load(self._best_checkpoint_path, map_location=self._device))

    def load_latest_checkpoint(self) -> int:
        return self._load_checkpoint(torch.load(self._latest_checkpoint_path, map_location=self._device))

    def load_imip_checkpoint(self, path: str):
        checkpoint_dict = torch.load(path, map_location=self._device)
        self._epi_point_net.imip_net.load_state_dict(checkpoint_dict["network_state_dict"])
        return

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

    # ################ TRAINING ################

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
        # corr_img_1_imip_outs = torch.zeros_like(img_1_imip_outs)

        imip_loss, imip_loss_logs = self._imip_loss_func(
            img_1_imip_outs, corr_img_1_imip_outs,
            img_1_inlier_labels, img_1_outlier_labels
        )
        imip_loss_logs = {"imip_" + k: v for k, v in imip_loss_logs.items()}

        # Find the distance between the points which fit the ground truth fundamental matrix
        # and the epipolar lines generated by the fundamental matrices produced by the DFE network
        # Note that we cap the point-wise losses in order to promote training stability.
        epi_loss = torch.tensor(0., device=self._device)
        for f_mat in f_mats:
            # TODO: Why don't we normalize these F matrices?
            epi_loss += robust_symmetric_epipolar_distance(image_1_epi_points, image_2_epi_points, f_mat).mean()

        loss = imip_loss
        if img_1_inlier_labels.sum() >= 8:  # Only train the epi loss if there are enough inliers
            loss += epi_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        f_estimated = f_mats[-1]
        f_estimated = f_estimated / f_estimated[:, -1, -1]
        f_estimated_loss = symmetric_epipolar_distance(image_1_epi_points, image_2_epi_points, f_estimated).mean()

        loss_logs = {
            "loss": loss.detach(),
            "epi_loss": epi_loss.detach(),
            "f_estimated_loss": f_estimated_loss.detach(),
        }
        loss_logs.update(imip_loss_logs)

        return loss_logs

    def _train_pair(self, pair: CorrespondenceFundamentalMatrixPair,
                    t_board_writer: SummaryWriter, iteration: int) -> [torch.Tensor]:
        """
        Trains the PatchBatchEpiPointNet on a stereo pair with both correspondence and fundamental matrix information
        :param pair: A stereo pair with both correspondence and fundamental matrix information
        :param iteration: how many pairs have been processed so far after processing this pair
        :return: the loss_logs generated by processing the pair with image 2 as the origin image
        """
        # Convert numpy HW or HWC to torch CHW
        img_1 = load_image_for_torch(pair.image_1, self._device)
        img_2 = load_image_for_torch(pair.image_2, self._device)

        # Extract the anchor keypoints in each image with the IMIP network
        exclude_border_px = (self._epi_point_net.patch_size() - 1) // 2
        img_1_keypoints_xy, img_1_responses = self._epi_point_net.imip_net.extract_keypoints(img_1, exclude_border_px)
        img_2_keypoints_xy, img_2_responses = self._epi_point_net.imip_net.extract_keypoints(img_2, exclude_border_px)

        """
        img_1_responses = img_1_responses.unsqueeze(1)  # Convert CHW to N1HW
        img_2_responses = img_2_responses.unsqueeze(1)
        t_board_writer.add_images("training/Image 1 Responses", img_1_responses, iteration)
        t_board_writer.add_images("training/Image 2 Responses", img_2_responses, iteration)
        """

        # Find the true correspondences for each anchor keypoint
        # img_1_correspondences_xy are the correspondences in img_2 of the anchor keypoints
        #   in img_1 (img_1_keypoints_xy)
        # img_2_correspondences_xy are the correspondences in img_1 of the anchor keypoints
        #   in img_2 (img_2_keypoints_xy)
        (img_1_correspondences_xy, img_1_correspondences_mask,
         img_2_correspondences_xy, img_2_correspondences_mask) = ImipTrainer.find_correspondences(
            pair, img_1_keypoints_xy, img_2_keypoints_xy, exclude_border_px
        )

        """
        # Log out the anchors and true correspondences
        img_1_true_corr_img = ImipTrainer.create_correspondence_image(pair, img_1_keypoints_xy,
                                                                      img_1_correspondences_xy,
                                                                      img_1_correspondences_mask)
        img_2_true_corr_img = ImipTrainer.create_correspondence_image(pair, img_2_correspondences_xy,
                                                                      img_2_keypoints_xy,
                                                                      img_2_correspondences_mask)
        t_board_writer.add_image("training/Predicted Anchors in Image 1 and True Correspondences in Image 2",
                                       img_1_true_corr_img, iteration)
        t_board_writer.add_image("training/Predicted Anchors in Image 2 and True Correspondences in Image 1",
                                       img_2_true_corr_img, iteration)
        """

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
        t_board_writer.add_scalar("training/apparent inliers", apparent_inliers, iteration)
        t_board_writer.add_scalar("training/apparent outliers", apparent_outliers, iteration)

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

        """
        t_board_writer.add_images("training/Image 1 Anchor Patches", img_1_anchor_patches / 255,
                                        iteration)
        t_board_writer.add_images("training/Image 1 Correspondence Patches", img_1_corr_patches / 255,
                                        iteration)
        t_board_writer.add_images("training/Image 2 Anchor Patches", img_2_anchor_patches / 255,
                                        iteration)
        t_board_writer.add_images("training/Image 2 Correspondence Patches", img_2_corr_patches / 255,
                                        iteration)
        """

        # Generate points which fit the ground truth fundamental matrix
        pts_1_virt, pts_2_virt = pair.generate_virtual_points()
        pts_1_virt = torch.tensor(pts_1_virt, device=self._device, dtype=torch.float32,
                                  requires_grad=False).unsqueeze(0)
        pts_2_virt = torch.tensor(pts_2_virt, device=self._device, dtype=torch.float32,
                                  requires_grad=False).unsqueeze(0)

        # Train with image 1 as the origin image
        self._train_patches(
            img_1_anchor_patches, img_2_anchor_patches,
            img_1_keypoints_xy, img_2_keypoints_xy,
            img_1_inlier_labels, img_1_outlier_labels, img_1_corr_patches,
            pts_1_virt, pts_2_virt
        )

        # Train with image 2 as the origin image
        loss_logs = self._train_patches(
            img_2_anchor_patches, img_1_anchor_patches,
            img_2_keypoints_xy, img_1_keypoints_xy,
            img_2_inlier_labels, img_2_outlier_labels, img_2_corr_patches,
            pts_2_virt, pts_1_virt
        )

        for key in loss_logs:
            t_board_writer.add_scalar("training/" + key, loss_logs[key], iteration)

        return loss_logs["loss"]

    def _evaluate_dataset(self, dataset: torch.utils.data.Dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = torch.tensor([len(dataset)], device=self._device)
        total_apparent_inliers = torch.tensor([0], device=self._device)
        total_true_inliers = torch.tensor([0], device=self._device)
        total_f_err = torch.tensor([0.], device=self._device)
        for pair in dataset:
            pair: CorrespondenceFundamentalMatrixPair = pair
            image_1 = load_image_for_torch(pair.image_1, device=self._device)
            image_2 = load_image_for_torch(pair.image_2, device=self._device)

            f_est, image_1_keypoints_xy, image_2_keypoints_xy, _ = self._epi_point_net.find_fundamental_matrix(
                image_1, image_2
            )

            num_apparent_inliers, num_true_inliers = ImipTrainer.count_inliers(
                pair, image_1_keypoints_xy, image_2_keypoints_xy, self._inlier_radius
            )
            total_apparent_inliers = total_apparent_inliers + num_apparent_inliers
            total_true_inliers = total_true_inliers + num_true_inliers

            pts_1_virt, pts_2_virt = pair.generate_virtual_points()
            pts_1_virt = torch.tensor(pts_1_virt, device=self._device, dtype=torch.float32,
                                      requires_grad=False).unsqueeze(0)
            pts_2_virt = torch.tensor(pts_2_virt, device=self._device, dtype=torch.float32,
                                      requires_grad=False).unsqueeze(0)
            total_f_err += symmetric_epipolar_distance(pts_1_virt, pts_2_virt, f_est).mean()
        return total_apparent_inliers / num_samples, total_true_inliers / num_samples, total_f_err / num_samples

    def train(self, iterations: int, eval_frequency: int):
        if iterations % eval_frequency != 0:
            raise ValueError(
                "eval_frequency ({}) must evenly divide iterations ({})".format(eval_frequency, iterations))

        t_board_writer = torch.utils.tensorboard.SummaryWriter()
        self._epi_point_net.train(True)
        self._imip_loss_module.train(True)

        last_eval_time = time.time()
        for iteration in range(1, iterations + 1):
            curr_pair = self._next_pair()
            curr_loss = self._train_pair(curr_pair, t_board_writer, iteration)

            if iteration % eval_frequency == 0:
                train_apparent_inliers_avg, train_true_inliers_avg, train_f_est_err_avg = self._evaluate_dataset(
                    self._train_eval_dataset)
                eval_apparent_inliers_avg, eval_true_inliers_avg, eval_f_est_err_avg = self._evaluate_dataset(
                    self._eval_dataset)
                t_board_writer.add_scalar("training_evaluation/apparent inliers", train_apparent_inliers_avg,
                                          iteration)
                t_board_writer.add_scalar("training_evaluation/true inliers", train_true_inliers_avg, iteration)
                t_board_writer.add_scalar("training_evaluation/f estimated loss", train_f_est_err_avg, iteration)
                t_board_writer.add_scalar("evaluation/apparent inliers", eval_apparent_inliers_avg, iteration)
                t_board_writer.add_scalar("evaluation/true inliers", eval_true_inliers_avg, iteration)
                t_board_writer.add_scalar("evaluation/f estimated loss", eval_f_est_err_avg, iteration)

                save_info = {
                    "inlier_radius": self._inlier_radius,
                    "iteration": iteration,
                    "network_state_dict": self._epi_point_net.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "training/loss": curr_loss,
                    "seed": self._seed,
                }

                torch.save(save_info, self._latest_checkpoint_path)

                if eval_f_est_err_avg < self._best_eval_f_err_score:
                    self._best_eval_f_err_score = eval_f_est_err_avg
                    torch.save(save_info, self._best_checkpoint_path)

                curr_eval_time = time.time()
                iters_per_minute = eval_frequency / ((curr_eval_time - last_eval_time) / 60)
                t_board_writer.add_scalar("performance/iterations per minute", iters_per_minute, iteration)
                last_eval_time = curr_eval_time
