import itertools
import os.path
import random
import time
from typing import Iterator, Callable, Union, Iterable, Tuple, Dict, Optional

import cv2
import numpy as np
import torch.utils.data
import torch.utils.tensorboard
from torch.optim.optimizer import Optimizer as TorchOptimizer

import epipointnet.data.image
import epipointnet.data.pairs
import epipointnet.datasets.shuffle
import epipointnet.datasets.tum_mono
import epipointnet.imips_pytorch.losses.imips
import epipointnet.imips_pytorch.models.imips


class OHNMImipTrainer:

    def __init__(self,
                 network: epipointnet.imips_pytorch.models.imips.ImipNet,
                 loss: epipointnet.imips_pytorch.losses.imips.ImipLoss,
                 optimizer_factory: Callable[[Union[Iterable[torch.Tensor], dict]], TorchOptimizer],
                 train_dataset: torch.utils.data.Dataset,
                 eval_dataset: torch.utils.data.Dataset,
                 num_eval_samples: int,
                 save_directory: str,
                 device: Optional[Union[torch.device, str]] = None,
                 inlier_radius: Optional[int] = 3,
                 num_candidate_patches: Optional[int] = 2,
                 seed: Optional[int] = 0,
                 ):
        self._seed = seed
        np.random.seed(self._seed)
        torch.random.manual_seed(self._seed)
        random.seed(self._seed)

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._device = device

        self._network = network.to(device=self._device)
        self._loss_module = loss.to(device=self._device)
        self._loss = self._loss_module.forward_with_log_data
        self._optimizer = optimizer_factory(itertools.chain(self._network.parameters(), self._loss_module.parameters()))
        self._train_eval_dataset = epipointnet.datasets.shuffle.ShuffledDataset(train_dataset, num_eval_samples)
        self._train_dataset = self._train_eval_dataset
        self._train_dataset_iter = self._create_new_train_dataset_iterator()
        self._eval_dataset = epipointnet.datasets.shuffle.ShuffledDataset(eval_dataset, num_eval_samples)
        self._inlier_radius = inlier_radius
        self._num_candidate_patches = num_candidate_patches
        self._best_eval_inlier_score = torch.tensor([0], device=self._device)
        self._best_checkpoint_path = os.path.join(save_directory, "imips_best.chkpt")
        self._latest_checkpoint_path = os.path.join(save_directory, "imips_latest.chkpt")

    # ################ CHECKPOINTING ################

    def _load_checkpoint(self, checkpoint: dict) -> int:
        self._seed = checkpoint["seed"]
        np.random.seed(self._seed)
        torch.random.manual_seed(self._seed)
        random.seed(self._seed)

        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._network.load_state_dict(checkpoint["network_state_dict"])
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

    # ################ DATASET ITERATION ################

    def _create_new_train_dataset_iterator(self) -> Iterator[epipointnet.data.pairs.CorrespondencePair]:
        # reshuffle the original dataset_name and begin iterating one pair at a time
        shuffled_dataset = epipointnet.datasets.shuffle.ShuffledDataset(self._train_dataset)
        return iter(torch.utils.data.DataLoader(shuffled_dataset, batch_size=None, collate_fn=lambda x: x))

    def _next_pair(self) -> epipointnet.data.pairs.CorrespondencePair:
        try:
            return next(self._train_dataset_iter)
        except StopIteration:
            self._train_dataset_iter = self._create_new_train_dataset_iterator()
            return next(self._train_dataset_iter)

    # ################ TRAINING ################

    @staticmethod
    def find_correspondences(pair: epipointnet.data.pairs.CorrespondencePair,
                             keypoints_xy: torch.Tensor,
                             inverse: bool = False,
                             exclude_border_px: int = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        keypoints_np = keypoints_xy.cpu().numpy()
        packed_correspondences_xy, correspondences_indices = pair.correspondences(
            keypoints_np,
            inverse=inverse
        )

        unpacked_correspondences_xy = np.zeros(keypoints_np.shape, dtype=packed_correspondences_xy.dtype)
        unpacked_correspondences_xy[:, correspondences_indices] = packed_correspondences_xy

        correspondences_mask = np.zeros(keypoints_np.shape[1], dtype=np.bool)
        correspondences_mask[correspondences_indices] = True

        correspondences_xy = torch.tensor(unpacked_correspondences_xy, device=keypoints_xy.device)
        correspondences_mask = torch.tensor(correspondences_mask, device=keypoints_xy.device)

        # remove correspondences in border area
        image_shape = pair.image_1.shape if inverse else pair.image_2.shape
        correspondence_in_border = (
                (correspondences_xy < exclude_border_px).sum(0).to(torch.bool) |
                ((image_shape[1] - exclude_border_px) <= correspondences_xy[0, :]) |
                ((image_shape[0] - exclude_border_px) <= correspondences_xy[1, :])
        )
        correspondences_xy[:, correspondence_in_border] = 0
        correspondences_mask[correspondence_in_border] = False
        return correspondences_xy, correspondences_mask

    @staticmethod
    def sort_candidates_and_generate_labels(
            img_1_kps: torch.Tensor, img_1_corrs: torch.Tensor, img_1_corrs_mask: torch.Tensor,
            img_2_kps: torch.Tensor, img_2_corrs: torch.Tensor, img_2_corrs_mask: torch.Tensor,
            inlier_radius: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # find the distance between the kp candidates and the correspondence for each heat map (cxkxk)
        img_1_kp_distances = torch.cdist(
            img_1_kps.permute(1, 2, 0),
            img_1_corrs.permute(1, 2, 0)
        )  # cxkxk
        img_1_match_dist_mask = ~img_1_corrs_mask.unsqueeze(1).repeat(1, img_1_corrs_mask.shape[-1], 1)
        img_1_kp_distances[img_1_match_dist_mask] = float("inf")

        img_2_kp_distances = torch.cdist(
            img_2_kps.permute(1, 2, 0),
            img_2_corrs.permute(1, 2, 0)
        )  # cxkxk
        img_2_match_dist_mask = ~img_2_corrs_mask.unsqueeze(1).repeat(1, img_2_corrs_mask.shape[-1], 1)
        img_2_kp_distances[img_2_match_dist_mask] = float("inf")

        match_distances = (img_1_kp_distances + img_2_kp_distances.permute(0, 2, 1)) / 2.0  # cxkxk

        best_match_scores, best_match_linear_idx = match_distances.flatten(1).min(dim=-1)  # c, c
        img_1_best_match = best_match_linear_idx // match_distances.shape[-1]  # c
        img_2_best_match = best_match_linear_idx % match_distances.shape[-1]  # c

        # implicit & img_1_corrs_mask[:, img_1_best_match] & img_2_corrs_mask[:, img_2_best_match]
        inlier_channels_by_best_match = (best_match_scores < inlier_radius)
        outlier_channels_by_best_match = (best_match_scores >= inlier_radius) & (best_match_scores != float('inf'))

        # implicit & img_1_corrs_mask[:, 0]
        img_1_inlier_channels_by_max = (img_1_kp_distances[:, 0, 0] < inlier_radius)
        img_1_outlier_channels_by_max = (
                (img_1_kp_distances[:, 0, 0] > inlier_radius) & (img_1_kp_distances[:, 0, 0] != float('inf'))
        )

        # implicit & img_2_corrs_mask[:, 0]
        img_2_inlier_channels_by_max = (img_2_kp_distances[:, 0, 0] < inlier_radius)
        img_2_outlier_channels_by_max = (
                (img_2_kp_distances[:, 0, 0] > inlier_radius) & (img_2_kp_distances[:, 0, 0] != float('inf'))
        )

        inlier_channels_by_max = img_1_inlier_channels_by_max & img_2_inlier_channels_by_max
        outlier_channels_by_max = img_1_outlier_channels_by_max | img_2_outlier_channels_by_max

        # swap best matches to the front of each group of patches
        temp = img_1_kps[:, :, 0]
        img_1_kps[:, :, 0] = img_1_kps[:, torch.arange(img_1_kps.shape[1]), img_1_best_match]
        img_1_kps[:, torch.arange(img_1_kps.shape[1]), img_1_best_match] = temp

        temp = img_2_kps[:, :, 0]
        img_2_kps[:, :, 0] = img_2_kps[:, torch.arange(img_2_kps.shape[1]), img_2_best_match]
        img_2_kps[:, torch.arange(img_2_kps.shape[1]), img_2_best_match] = temp

        return (
            img_1_kps, img_2_kps, inlier_channels_by_max, outlier_channels_by_max,
            inlier_channels_by_best_match, outlier_channels_by_best_match
        )

    @staticmethod
    def create_correspondence_image(pair: epipointnet.data.pairs.CorrespondencePair, image_1_pixels_xy: torch.Tensor,
                                    image_2_pixels_xy: torch.Tensor,
                                    correspondence_mask: torch.Tensor) -> torch.Tensor:
        image_1_pixels_xy = image_1_pixels_xy.cpu()
        image_2_pixels_xy = image_2_pixels_xy.cpu()
        correspondence_mask = correspondence_mask.cpu()

        image_1_keypoints = [cv2.KeyPoint(image_1_pixels_xy[0][i], image_1_pixels_xy[1][i], 1) for i in
                             range(image_1_pixels_xy.shape[1])]
        image_2_keypoints = [cv2.KeyPoint(image_2_pixels_xy[0][i], image_2_pixels_xy[1][i], 1) for i in
                             range(image_2_pixels_xy.shape[1])]

        matches = np.arange(correspondence_mask.shape[0])
        matches = np.vstack((matches, matches))
        matches = matches[:, correspondence_mask]
        matches = [cv2.DMatch(matches[0][i], matches[1][i], 0.0) for i in range(matches.shape[1])]

        corr_img = cv2.drawMatches(pair.image_1, image_1_keypoints, pair.image_2, image_2_keypoints, matches, None)
        return torch.tensor(corr_img).permute((2, 0, 1))

    @staticmethod
    def image_to_patch_batch(image: torch.Tensor, keypoints_xy: torch.Tensor, diameter: int) -> torch.Tensor:
        if diameter % 2 != 1:
            raise ValueError("diameter must be odd")
        assert len(keypoints_xy.shape) == 2 and keypoints_xy.shape[0] == 2
        radius = (diameter - 1) // 2
        keypoints_xy = keypoints_xy.to(torch.int)
        batch = torch.zeros((keypoints_xy.shape[1], image.shape[0], diameter, diameter), device=keypoints_xy.device)
        for point_idx in range(keypoints_xy.shape[1]):
            keypoint_x = keypoints_xy[0, point_idx]
            keypoint_y = keypoints_xy[1, point_idx]
            batch[point_idx, :, :, :] = image[
                                        :,
                                        keypoint_y - radius: keypoint_y + radius + 1,
                                        keypoint_x - radius: keypoint_x + radius + 1
                                        ]
        return batch

    def _train_patches(self, maximizer_patches: torch.Tensor, correspondence_patches: torch.Tensor,
                       inlier_labels: torch.Tensor,
                       outlier_labels: torch.Tensor) -> Dict[str, torch.Tensor]:

        # comments assume maximizer_patches and correspondence_patches are both pulled from image 1 in a pair

        # maximizer_patches: BNxCxHxW
        # correspondence_patches: BxCxHxW
        # inlier_labels: B
        # outlier_labels: B

        # ensure the number of channels in the patches matches the input channels of the network
        assert (len(maximizer_patches.shape) == len(correspondence_patches.shape) == 4 and
                maximizer_patches.shape[1] == correspondence_patches.shape[1] == self._network.input_channels())

        # ensure the number of labels equals the number of output channels
        assert (len(inlier_labels.shape) == len(outlier_labels.shape) == 1 and
                inlier_labels.shape[0] == outlier_labels.shape[0] == self._network.output_channels())

        # ensure number of labels/ maximizer patches is a multiple of the number of corr. patches/ output channels
        assert (maximizer_patches.shape[0] % correspondence_patches.shape[0] == 0)
        assert (maximizer_patches.shape[0] % self._network.output_channels() == 0)

        # ensure the patches match the size of the receptive field
        assert (self._network.receptive_field_diameter() == maximizer_patches.shape[2] == maximizer_patches.shape[3] ==
                correspondence_patches.shape[2] == correspondence_patches.shape[3])

        maximizer_outputs: torch.Tensor = self._network(maximizer_patches, False)
        correspondence_outputs: torch.Tensor = self._network(correspondence_patches, False)

        loss, loss_logs = self._loss(maximizer_outputs, correspondence_outputs, inlier_labels, outlier_labels)

        # run the optimizer with the loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss_logs

    def _train_pair(self, pair: epipointnet.data.pairs.CorrespondencePair,
                    t_board_writer: torch.utils.tensorboard.SummaryWriter, iteration: int) -> torch.Tensor:

        img_1 = epipointnet.data.image.load_image_for_torch(pair.image_1, self._device)  # c x h x w
        img_2 = epipointnet.data.image.load_image_for_torch(pair.image_2, self._device)

        img_1_kp_candidates, _ = self._network.extract_top_k_keypoints(img_1, self._num_candidate_patches)  # 2 x c x k
        img_2_kp_candidates, _ = self._network.extract_top_k_keypoints(img_2, self._num_candidate_patches)

        img_1_correspondences, img_1_correspondences_mask = self.find_correspondences(
            pair, img_2_kp_candidates.flatten(1), inverse=True,
            exclude_border_px=(self._network.receptive_field_diameter() - 1) // 2
        )  # 2 x ck, ck
        img_1_correspondences = img_1_correspondences.reshape_as(img_1_kp_candidates)  # 2 x c x k
        img_1_correspondences_mask = img_1_correspondences_mask.reshape(-1, self._num_candidate_patches)  # c x k

        img_2_correspondences, img_2_correspondences_mask = self.find_correspondences(
            pair, img_1_kp_candidates.flatten(1), inverse=False,
            exclude_border_px=(self._network.receptive_field_diameter() - 1) // 2
        )
        img_2_correspondences = img_2_correspondences.reshape_as(img_2_kp_candidates)  # 2 x c x k
        img_2_correspondences_mask = img_2_correspondences_mask.reshape(-1, self._num_candidate_patches)  # c x k

        (img_1_kp_candidates, img_2_kp_candidates, apparent_inliers, apparent_outliers,
         inlier_channels_by_best_match, outlier_channels_by_best_match) = self.sort_candidates_and_generate_labels(
            img_1_kp_candidates, img_1_correspondences, img_1_correspondences_mask,
            img_2_kp_candidates, img_2_correspondences, img_2_correspondences_mask, self._inlier_radius
        )
        t_board_writer.add_scalar("training/apparent inliers", apparent_inliers.sum(), iteration)
        t_board_writer.add_scalar("training/apparent outliers", apparent_outliers.sum(), iteration)

        t_board_writer.add_scalar("training/apparent inliers (best match)", inlier_channels_by_best_match.sum(),
                                  iteration)
        t_board_writer.add_scalar("training/apparent outliers (best match)", outlier_channels_by_best_match.sum(),
                                  iteration)

        patch_diameter = self._network.receptive_field_diameter()
        kp_patches = OHNMImipTrainer.image_to_patch_batch(
            img_1, img_1_kp_candidates.flatten(1),
            patch_diameter
        )
        corr_patches = torch.zeros(
            img_1_correspondences.shape[1], img_1.shape[0], patch_diameter, patch_diameter,
            dtype=kp_patches.dtype,
            device=self._device
        )
        corr_patches[img_1_correspondences_mask[:, 0], :, :, :] = OHNMImipTrainer.image_to_patch_batch(
            img_1, img_1_correspondences[:, img_1_correspondences_mask[:, 0], 0], patch_diameter
        )

        self._train_patches(
            kp_patches, corr_patches, inlier_channels_by_best_match, outlier_channels_by_best_match
        )

        kp_patches = OHNMImipTrainer.image_to_patch_batch(
            img_2, img_2_kp_candidates.flatten(1),
            patch_diameter
        )
        corr_patches = torch.zeros(
            img_2_correspondences.shape[1], img_2.shape[0], patch_diameter, patch_diameter,
            dtype=kp_patches.dtype,
            device=self._device
        )
        corr_patches[img_2_correspondences_mask[:, 0], :, :, :] = OHNMImipTrainer.image_to_patch_batch(
            img_2, img_2_correspondences[:, img_2_correspondences_mask[:, 0], 0], patch_diameter
        )

        loss_logs = self._train_patches(
            kp_patches, corr_patches, inlier_channels_by_best_match, outlier_channels_by_best_match
        )

        for key in loss_logs:
            if loss_logs[key] is not None:
                t_board_writer.add_scalar("training/" + key, loss_logs[key], iteration)

        return loss_logs["loss"]

    @staticmethod
    def count_inliers(pair: epipointnet.data.pairs.CorrespondencePair,
                      img_1_kp_candidates: torch.Tensor, img_2_kp_candidates: torch.Tensor,
                      inlier_radius: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_1_correspondences, img_1_correspondences_mask = OHNMImipTrainer.find_correspondences(
            pair, img_2_kp_candidates.flatten(1), inverse=True,
        )  # 2 x ck, ck
        img_1_correspondences = img_1_correspondences.reshape_as(img_1_kp_candidates)  # 2 x c x k
        img_1_correspondences_mask = img_1_correspondences_mask.reshape(-1, img_1_correspondences.shape[-1])  # c x k

        img_2_correspondences, img_2_correspondences_mask = OHNMImipTrainer.find_correspondences(
            pair, img_1_kp_candidates.flatten(1), inverse=False,
        )
        img_2_correspondences = img_2_correspondences.reshape_as(img_2_kp_candidates)  # 2 x c x k
        img_2_correspondences_mask = img_2_correspondences_mask.reshape(-1, img_1_correspondences.shape[-1])  # c x k

        (_, _,
         apparent_inliers_by_max, apparent_outliers_by_max,
         inlier_channels_by_best_match,
         outlier_channels_by_best_match) = OHNMImipTrainer.sort_candidates_and_generate_labels(
            img_1_kp_candidates, img_1_correspondences, img_1_correspondences_mask,
            img_2_kp_candidates, img_2_correspondences, img_2_correspondences_mask, inlier_radius
        )

        num_apparent_inliers_by_max = apparent_inliers_by_max.sum()
        num_apparent_inliers_by_best_match = inlier_channels_by_best_match.sum()

        apparent_inlier_img_1_keypoints_xy = img_1_kp_candidates[:, apparent_inliers_by_max, 0]
        num_true_inliers_by_max = torch.zeros([1], device=img_1_kp_candidates.device)
        if num_apparent_inliers_by_max > 0:
            max_inlier_distance = torch.tensor([inlier_radius], device=img_1_kp_candidates.device)
            unique_inlier_img_1_keypoints_xy = apparent_inlier_img_1_keypoints_xy[:, 0:1]
            num_true_inliers_by_max += 1  # on Apr 22nd 2020, an off by one error was discovered +1, to old results
            for i in range(1, int(num_apparent_inliers_by_max)):
                test_inlier = apparent_inlier_img_1_keypoints_xy[:, i:i + 1]
                if (torch.norm(unique_inlier_img_1_keypoints_xy - test_inlier, p=2, dim=0) > max_inlier_distance).all():
                    unique_inlier_img_1_keypoints_xy = torch.cat((unique_inlier_img_1_keypoints_xy, test_inlier), dim=1)
                    num_true_inliers_by_max += 1

        return num_apparent_inliers_by_max, num_true_inliers_by_max, num_apparent_inliers_by_best_match

    def _evaluate_dataset(self, dataset: torch.utils.data.Dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = torch.tensor([len(dataset)], device=self._device)
        total_apparent_inliers = torch.tensor([0], device=self._device)
        total_true_inliers = torch.tensor([0], device=self._device)
        total_inliers_by_best_match = torch.tensor([0], device=self._device)
        for pair in dataset:
            pair: epipointnet.data.pairs.CorrespondencePair = pair
            img_1 = epipointnet.data.image.load_image_for_torch(pair.image_1, device=self._device)
            img_2 = epipointnet.data.image.load_image_for_torch(pair.image_2, device=self._device)
            img_1_kp_candidates, _ = self._network.extract_top_k_keypoints(img_1, self._num_candidate_patches)
            img_2_kp_candidates, _ = self._network.extract_top_k_keypoints(img_2, self._num_candidate_patches)

            num_apparent_inliers, num_true_inliers, best_match = OHNMImipTrainer.count_inliers(
                pair, img_1_kp_candidates, img_2_kp_candidates, self._inlier_radius
            )
            total_apparent_inliers = total_apparent_inliers + num_apparent_inliers
            total_true_inliers = total_true_inliers + num_true_inliers
            total_inliers_by_best_match = total_inliers_by_best_match + best_match
        return (total_apparent_inliers / num_samples, total_true_inliers / num_samples,
                total_inliers_by_best_match / num_samples)

    def train(self, iterations: int, eval_frequency: int):
        if iterations % eval_frequency != 0:
            raise ValueError(
                "eval_frequency ({}) must evenly divide iterations ({})".format(eval_frequency, iterations))

        t_board_path = os.path.join(".", "runs", os.path.basename(os.path.dirname(self._best_checkpoint_path)))
        t_board_writer = torch.utils.tensorboard.SummaryWriter(log_dir=t_board_path)

        self._network.train(True)
        self._loss_module.train(True)

        last_eval_time = time.time()
        for iteration in range(1, iterations + 1):
            curr_loss = self._train_pair(self._next_pair(), t_board_writer, iteration)
            if iteration % eval_frequency == 0:
                train_apparent_inliers_avg, train_true_inliers_avg, train_inliers_by_best_match_avg = \
                    self._evaluate_dataset(self._train_eval_dataset)
                eval_apparent_inliers_avg, eval_true_inliers_avg, eval_inliers_by_best_match_avg = \
                    self._evaluate_dataset(self._eval_dataset)
                t_board_writer.add_scalar("training_evaluation/apparent inliers", train_apparent_inliers_avg,
                                          iteration)
                t_board_writer.add_scalar("training_evaluation/true inliers", train_true_inliers_avg, iteration)
                t_board_writer.add_scalar(
                    "training_evaluation/apparent inliers (best match)", train_inliers_by_best_match_avg, iteration
                )
                t_board_writer.add_scalar("evaluation/apparent inliers", eval_apparent_inliers_avg, iteration)
                t_board_writer.add_scalar("evaluation/true inliers", eval_true_inliers_avg, iteration)
                t_board_writer.add_scalar(
                    "evaluation/apparent inliers (best match)", eval_inliers_by_best_match_avg, iteration
                )
                save_info = {
                    "inlier_radius": self._inlier_radius,
                    "iteration": iteration,
                    "network_state_dict": self._network.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "training/loss": curr_loss,
                    "training_evaluation/apparent inliers": train_apparent_inliers_avg,
                    "training_evaluation/true inliers": train_true_inliers_avg,
                    "training_evaluation/apparent inliers (best match)": train_inliers_by_best_match_avg,
                    "evaluation/apparent inliers": eval_apparent_inliers_avg,
                    "evaluation/true inliers": eval_true_inliers_avg,
                    "evaluation/apparent inliers (best match)": eval_inliers_by_best_match_avg,
                    "seed": self._seed,
                }

                torch.save(save_info, self._latest_checkpoint_path)

                if eval_true_inliers_avg > self._best_eval_inlier_score:
                    self._best_eval_inlier_score = eval_true_inliers_avg
                    torch.save(save_info, self._best_checkpoint_path)

                curr_eval_time = time.time()
                iters_per_minute = eval_frequency / ((curr_eval_time - last_eval_time) / 60)
                t_board_writer.add_scalar("performance/iterations per minute", iters_per_minute, iteration)
                last_eval_time = curr_eval_time
