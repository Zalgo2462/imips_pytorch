import os.path
import time
from typing import Iterator, Callable, Union, Iterable, Tuple

import numpy as np
import torch.utils.data
import torch.utils.tensorboard
from torch.optim.optimizer import Optimizer as TorchOptimizer

import epipolar_nn.dataloaders.image
import epipolar_nn.dataloaders.pair
import epipolar_nn.dataloaders.random
import epipolar_nn.dataloaders.tum
import epipolar_nn.imips_pytorch.networks.convnet
import epipolar_nn.imips_pytorch.networks.imips


class ImipsTrainer:

    def __init__(self,
                 network: epipolar_nn.imips_pytorch.networks.imips.ImipsNet,
                 optimizer_factory: Callable[[Union[Iterable[torch.Tensor], dict]], TorchOptimizer],
                 train_dataset: torch.utils.data.Dataset,
                 eval_dataset: torch.utils.data.Dataset,
                 num_eval_samples: int,
                 save_directory: str,
                 inlier_radius: int = 3
                 ):
        self._network = network
        self._optimizer = optimizer_factory(self._network.parameters())
        self._train_dataset = train_dataset
        self._train_dataset_iter = self._create_new_train_dataset_iterator()
        self._train_eval_dataset = epipolar_nn.dataloaders.random.ShuffledDataset(train_dataset, num_eval_samples)
        self._eval_dataset = epipolar_nn.dataloaders.random.ShuffledDataset(eval_dataset, num_eval_samples)
        self._inlier_radius = inlier_radius
        self._best_eval_inlier_score = torch.tensor([0], device="cuda")
        self._t_board_writer = torch.utils.tensorboard.SummaryWriter()
        self._best_checkpoint_path = os.path.join(save_directory, "imips_best.chkpt")
        self._latest_checkpoint_path = os.path.join(save_directory, "imips_latest.chkpt")

    def _create_new_train_dataset_iterator(self) -> Iterator[epipolar_nn.dataloaders.pair.StereoPair]:
        # reshuffle the original dataset and begin iterating one pair at a time
        shuffled_dataset = epipolar_nn.dataloaders.random.ShuffledDataset(self._train_dataset)
        return iter(torch.utils.data.DataLoader(shuffled_dataset, batch_size=None, collate_fn=lambda x: x))

    def _next_pair(self) -> epipolar_nn.dataloaders.pair.StereoPair:
        try:
            return next(self._train_dataset_iter)
        except StopIteration:
            self._train_dataset_iter = self._create_new_train_dataset_iterator()
            return next(self._train_dataset_iter)

    def _train_patches(self, maximizer_patches: torch.Tensor, correspondence_patches: torch.Tensor,
                       inlier_labels: torch.Tensor,
                       outlier_labels: torch.Tensor) -> torch.Tensor:

        # comments assume maximizer_patches and correspondence_patches are both pulled from image 1 in a pair

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

        # maximizer_patches: BxCxHxW
        # correspondence_patches: BxCxHxW
        # inlier_labels: B
        # outlier_labels: B

        # ensure the number of channels in the patches matches the input channels of the network
        assert len(maximizer_patches.shape) == len(correspondence_patches.shape) == 4 and \
               maximizer_patches.shape[1] == correspondence_patches.shape[1] == self._network.input_channels()

        # ensure the number of input patches matches the number of output channels
        assert len(inlier_labels.shape) == len(outlier_labels.shape) == 1 and inlier_labels.shape[0] == \
               outlier_labels.shape[0] == maximizer_patches.shape[0] == correspondence_patches.shape[0] == \
               self._network.output_channels()

        # ensure the patches match the size of the receptive field
        assert self._network.receptive_field_diameter() == maximizer_patches.shape[2] == maximizer_patches.shape[3] == \
               correspondence_patches.shape[2] == correspondence_patches.shape[3]

        # maximizer_outputs: BxCx1x1 where B == C
        # correspondence_outputs: BxCx1x1 where B == C
        maximizer_outputs: torch.Tensor = self._network(maximizer_patches, False)
        correspondence_outputs: torch.Tensor = self._network(correspondence_patches, False)

        assert maximizer_outputs.shape[0] == maximizer_outputs.shape[1] == \
               correspondence_outputs.shape[0] == correspondence_outputs.shape[1]

        assert maximizer_outputs.shape[2] == maximizer_outputs.shape[3] == \
               correspondence_outputs.shape[2] == correspondence_outputs.shape[3] == 1

        maximizer_outputs = maximizer_outputs.squeeze()  # BxCx1x1 -> BxC
        correspondence_outputs = correspondence_outputs.squeeze()  # BxCx1x1 -> BxC

        # The goal is to maximize each channel's response to it's patch in image 1 which corresponds
        # with it's maximizing patch in image 2. If the patch which maximizes a channel's response in
        # image 1 is within a given radius of the patch in image 1 which corresponds
        # with the channel's maximizing patch in image 2, the channel is assigned a loss of 0.
        # Otherwise, if the maximizing patch for a channel is outside of this radius, the channel's loss is
        # set to maximize the channel's response to the patch in image 1 which corresponds
        # with the channel's maximizing patch in image 2.
        #
        # Research note: why do we allow the maximum response to be within a radius? Why complicate the loss
        # this way? Maybe try always maximizing each channel's response to the patch in image 1 which
        # corresponds with the channel's maximizing patch in image 2.

        # grabs the outlier responses where the batch index and channel index align.
        aligned_outlier_index = torch.diag(outlier_labels)

        aligned_outlier_correspondence_scores = correspondence_outputs[aligned_outlier_index]
        assert len(aligned_outlier_correspondence_scores.shape) == 1

        # A lower response to a channel's patch in image 1 which corresponds with it's maximizing patch in image 2
        # will lead to a higher loss. This is called correspondence loss by imips
        outlier_correspondence_loss = torch.sum(-1 * torch.log(aligned_outlier_correspondence_scores))

        # If a channel's maximum response is outside of a given radius about the target correspondence site, the
        # channel's response to it's maximizing patch in image 1 is minimized.
        aligned_outlier_maximizer_scores = maximizer_outputs[aligned_outlier_index]

        # A higher response to a channel's maximizing patch in image 1 will lead to a
        # higher loss for a channel which attains its maximum outside of a given radius
        # about it's target correspondence site. This is called outlier loss by imips.
        outlier_maximizer_loss = torch.sum(-1 * torch.log(-1 * aligned_outlier_maximizer_scores + 1))

        outlier_loss = outlier_correspondence_loss + outlier_maximizer_loss

        # If a channel's maximum response is inside of a given radius about the target correspondence site, the
        # chanel's response to it's maximizing patch in image 1 is maximized.

        # grabs the inlier responses where the batch index and channel index align.
        aligned_inlier_index = torch.diag(inlier_labels)

        aligned_inlier_maximizer_scores = maximizer_outputs[aligned_inlier_index]
        assert len(aligned_inlier_maximizer_scores.shape) == 1

        # A lower response to a channel's maximizing patch in image 1 wil lead to
        # a higher loss for a channel which attains its maximum inside of a given radius
        # about it's target correspondence site. This is called inlier_loss by imips.
        inlier_loss = torch.sum(-1 * torch.log(aligned_inlier_maximizer_scores))

        # Finally, if a channel attains its maximum response inside of a given radius
        # about it's target correspondence site, the responses of all the other channels
        # to it's maximizing patch are minimized. (Kind of...)
        # equivalent: inlier_labels.unsqueeze(1).repeat(1, inlier_labels.shape[0]) - inlier_labels.diag()
        unaligned_inlier_index = aligned_inlier_index ^ inlier_labels.unsqueeze(1)
        unaligned_inlier_maximizer_scores = maximizer_outputs[unaligned_inlier_index]

        # imips just adds the unaligned scores to the loss directly
        loss = outlier_loss + inlier_loss + torch.sum(unaligned_inlier_maximizer_scores)

        # run the optimizer with the loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss

    def _train_pair(self, pair: epipolar_nn.dataloaders.pair.StereoPair) -> torch.Tensor:
        image_1 = epipolar_nn.dataloaders.image.load_image_for_torch(pair.image_1)
        image_2 = epipolar_nn.dataloaders.image.load_image_for_torch(pair.image_2)
        image_1_keypoints = self._network.extract_keypoints(image_1)
        image_2_keypoints = self._network.extract_keypoints(image_2)

        image_1_anchor_patches, image_1_corr_patches, image_1_inlier_labels, image_1_outlier_labels, \
        image_2_anchor_patches, image_2_corr_patches, image_2_inlier_labels, image_2_outlier_labels = \
            _generate_training_patches(
                pair, image_1_keypoints,
                image_2_keypoints,
                self._network.receptive_field_diameter(),
                self._inlier_radius,
            )

        self._train_patches(
            image_1_anchor_patches, image_1_corr_patches, image_1_inlier_labels, image_1_outlier_labels
        )
        loss = self._train_patches(
            image_2_anchor_patches, image_2_corr_patches, image_2_inlier_labels, image_2_outlier_labels
        )
        return loss

    def _evaluate_dataset(self, dataset: torch.utils.data.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        num_samples = torch.tensor([len(dataset)], device="cuda")
        total_apparent_inliers = torch.tensor([0], device="cuda")
        total_true_inliers = torch.tensor([0], device="cuda")
        for pair in dataset:
            pair: epipolar_nn.dataloaders.pair.StereoPair = pair
            image_1 = epipolar_nn.dataloaders.image.load_image_for_torch(pair.image_1)
            image_2 = epipolar_nn.dataloaders.image.load_image_for_torch(pair.image_2)
            image_1_keypoints_xy = self._network.extract_keypoints(image_1)
            image_2_keypoints_xy = self._network.extract_keypoints(image_2)

            num_apparent_inliers, num_true_inliers = _count_inliers(
                pair, image_1_keypoints_xy, image_2_keypoints_xy, self._inlier_radius
            )
            total_apparent_inliers = total_apparent_inliers + num_apparent_inliers
            total_true_inliers = total_true_inliers + num_true_inliers
        return total_apparent_inliers / num_samples, total_true_inliers / num_samples

    def _load_checkpoint(self, checkpoint: dict):
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._network.load_state_dict(checkpoint["network_state_dict"])
        self._inlier_radius = checkpoint["inlier_radius"]
        for i in range(0, checkpoint["iteration"]):
            self._next_pair()

    def load_best_checkpoint(self):
        self._load_checkpoint(torch.load(self._best_checkpoint_path))

    def load_latest_checkpoint(self):
        self._load_checkpoint(torch.load(self._latest_checkpoint_path))

    def train(self, iterations: int, eval_frequency: int):
        if iterations % eval_frequency != 0:
            raise ValueError(
                "eval_frequency ({}) must evenly divide iterations ({})".format(eval_frequency, iterations))

        last_eval_time = time.time()
        for iteration in range(1, iterations + 1):
            curr_pair = self._next_pair()
            curr_loss = self._train_pair(curr_pair)
            self._t_board_writer.add_scalar("training/loss", curr_loss, iteration)
            if iteration % eval_frequency == 0:
                train_apparent_inliers_avg, train_true_inliers_avg = self._evaluate_dataset(self._train_eval_dataset)
                eval_apparent_inliers_avg, eval_true_inliers_avg = self._evaluate_dataset(self._eval_dataset)
                self._t_board_writer.add_scalar("training/apparent inliers", train_apparent_inliers_avg, iteration)
                self._t_board_writer.add_scalar("training/true inliers", train_true_inliers_avg, iteration)
                self._t_board_writer.add_scalar("evaluation/apparent inliers", eval_apparent_inliers_avg, iteration)
                self._t_board_writer.add_scalar("evaluation/true inliers", eval_true_inliers_avg, iteration)
                save_info = {
                    "inlier_radius": self._inlier_radius,
                    "iteration": iteration,
                    "network_state_dict": self._network.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "training/loss": curr_loss,
                    "training/apparent inliers": train_apparent_inliers_avg,
                    "training/true inliers": train_true_inliers_avg,
                    "evaluation/apparent inliers": eval_apparent_inliers_avg,
                    "evaluation/true inliers": eval_true_inliers_avg,
                }

                torch.save(save_info, self._latest_checkpoint_path)

                if eval_true_inliers_avg > self._best_eval_inlier_score:
                    self._best_eval_inlier_score = eval_true_inliers_avg
                    torch.save(save_info, self._best_checkpoint_path)

                curr_eval_time = time.time()
                iters_per_minute = eval_frequency / ((curr_eval_time - last_eval_time) / 60)
                self._t_board_writer.add_scalar("performance/iterations per minute", iters_per_minute, iteration)
                last_eval_time = curr_eval_time


def _generate_training_patches(
        pair: epipolar_nn.dataloaders.pair.StereoPair,
        img_1_keypoints_xy: torch.Tensor, img_2_keypoints_xy: torch.Tensor,
        patch_diameter: int, inlier_distance: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a training pair and the keypoints detected by the network for each image,
    return the training patches for the predicted anchors and true correspondences,
    as well as the inlier/outlier labels for each image.

    The patch diameter controls the size of the returned patches. This should
    correspond with the number of convolutions in the network. i.e.
    patch_diameter = num_conv_layers * 2 + 1 if each layer is a 3x3 convolution.

    Inlier_distance controls the labelling of inliers. The predicted anchor in one image
    and the true correspondence of the anchor predicted in the other image must be
    this close to be considered an inlier. If there exists a valid correspondence,
    but the distance from the anchor to the correspondence is beyond this limit,
    the prediction is labelled as an outlier.

    Returns an eight-tuple.
    The first four entries corresponds to the first image in the pair, and the last four entries
    corresponds to the second image.
    Each set has the form: (anchor patches, correspondence patches, inlier labels, outlier labels)
    Anchor patches and correspondence patches are tensors of the form BxCxHxW
    """
    img_1_correspondences_xy, img_1_correspondences_mask, \
    img_2_correspondences_xy, img_2_correspondences_mask = _find_correspondences(
        pair, img_1_keypoints_xy, img_2_keypoints_xy
    )

    img_1_inliers, img_1_outliers, img_2_inliers, img_2_outliers = _label_inliers_outliers(
        img_1_keypoints_xy, img_1_correspondences_xy, img_1_correspondences_mask,
        img_2_keypoints_xy, img_2_correspondences_xy, img_2_correspondences_mask,
        inlier_distance
    )

    img_1_kp_patches = _image_to_patch_batch(pair.image_1, img_1_keypoints_xy, patch_diameter)
    img_1_corr_patches = torch.zeros_like(img_1_kp_patches, device="cuda")
    img_1_corr_patches[img_2_correspondences_mask, :, :, :] = _image_to_patch_batch(
        pair.image_1,
        img_2_correspondences_xy[:, img_2_correspondences_mask],
        patch_diameter
    )
    img_2_kp_patches = _image_to_patch_batch(pair.image_2, img_2_keypoints_xy, patch_diameter)
    img_2_corr_patches = torch.zeros_like(img_2_kp_patches, device="cuda")
    img_2_corr_patches[img_1_correspondences_mask, :, :, :] = _image_to_patch_batch(
        pair.image_2,
        img_1_correspondences_xy[:, img_1_correspondences_mask],
        patch_diameter
    )

    # The results are returned in in two batches, one for each image that was subsampled.
    # This is a holdover from the original work. It might make more sense to
    # swap the kp and corr patches around in the future.
    # img 1_kp_patches come from the anchor points detected in img 1
    # img 1 corr patches come from the real correspondences of the anchor points detected in img 2
    # img 2 kp patches come from the anchor points detected in img 2
    # img 2 corr patches come from the real correspondences of the anchor points detected in img 1
    # img_1_inliers is true at an index when there is a valid, true correspondence in img_1 for the anchor in img_2
    #   and the anchor in img_1 is close to the true correspondence in img_1
    # img_2_inliers is true at an index when there is a valid, true correspondence in img_2 for the anchor in img_1
    #   and the anchor in img_2 is close to the true correspondence in img_2
    return img_1_kp_patches, img_1_corr_patches, img_1_inliers, img_1_outliers, \
           img_2_kp_patches, img_2_corr_patches, img_2_inliers, img_2_outliers


def _count_inliers(pair: epipolar_nn.dataloaders.pair.StereoPair,
                   img_1_keypoints_xy: torch.Tensor, img_2_keypoints_xy: torch.Tensor,
                   inlier_distance: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    img_1_correspondences_xy, img_1_correspondences_mask, \
    img_2_correspondences_xy, img_2_correspondences_mask = _find_correspondences(
        pair, img_1_keypoints_xy, img_2_keypoints_xy
    )

    img_1_inliers, img_1_outliers, img_2_inliers, img_2_outliers = _label_inliers_outliers(
        img_1_keypoints_xy, img_1_correspondences_xy, img_1_correspondences_mask,
        img_2_keypoints_xy, img_2_correspondences_xy, img_2_correspondences_mask,
        inlier_distance
    )

    apparent_inliers = img_1_inliers & img_2_inliers
    num_apparent_inliers = apparent_inliers.sum()
    apparent_inlier_img_1_keypoints_xy = img_1_keypoints_xy[:, apparent_inliers]

    num_true_inliers = torch.zeros(1, device="cuda")

    if num_apparent_inliers > 0:
        max_inlier_distance = torch.tensor([inlier_distance], device="cuda")
        unique_inlier_img_1_keypoints_xy = apparent_inlier_img_1_keypoints_xy[:, 0:1]
        for i in range(1, int(num_apparent_inliers)):
            test_inlier = apparent_inlier_img_1_keypoints_xy[:, i:i + 1]
            if (torch.norm(unique_inlier_img_1_keypoints_xy - test_inlier, p=2, dim=0) > max_inlier_distance).all():
                unique_inlier_img_1_keypoints_xy = torch.cat((unique_inlier_img_1_keypoints_xy, test_inlier), dim=1)
                num_true_inliers += 1

    return num_apparent_inliers, num_true_inliers


def _find_correspondences(pair: epipolar_nn.dataloaders.pair.StereoPair,
                          img_1_keypoints_xy: torch.Tensor,
                          img_2_keypoints_xy: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Convert keypoints to numpy data structures to compute the correspondences
    # The correspondences are calculated via numpy/ opencv, and therefore are likely
    # computed on the CPU
    img_1_keypoints_xy_np = img_1_keypoints_xy.cpu().numpy()
    img_2_keypoints_xy_np = img_2_keypoints_xy.cpu().numpy()

    img_1_packed_correspondences_xy, img_1_correspondences_indices = pair.correspondences(
        img_1_keypoints_xy_np,
        inverse=False
    )
    img_2_packed_correspondences_xy, img_2_correspondences_indices = pair.correspondences(
        img_2_keypoints_xy_np,
        inverse=True
    )

    # img_1_correspondences_xy is (zero, zero).T where img_1_correspondences_mask is False
    # unpack_correspondences returns torch tensors, this will move the data back to the GPU
    img_1_correspondences_xy, img_1_correspondences_mask = _unpack_correspondences(
        img_1_keypoints_xy_np,
        img_1_packed_correspondences_xy,
        img_1_correspondences_indices
    )
    img_2_correspondences_xy, img_2_correspondences_mask = _unpack_correspondences(
        img_2_keypoints_xy_np,
        img_2_packed_correspondences_xy,
        img_2_correspondences_indices,
    )

    return img_1_correspondences_xy, img_1_correspondences_mask, img_2_correspondences_xy, img_2_correspondences_mask


def _unpack_correspondences(keypoints_xy: np.ndarray, correspondences_xy: np.ndarray, correspondence_indices) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the correspondences from their packed form to their unpacked form.

    In the packed form, the items of correspondences_xy are mapped to keypoints_xy
    by the correspondence_indices array. Where correspondences_xy[i] matches
    keypoints_xy[correspondence_indicies[i]].

    In the unpacked form, the correspondences are matched to keypoints_xy by index alone.
    If no correspondence exists for a given point in keypoints_xy, the resulting
    correspondence is (zero, zero).T. The correspondence_mask is aligned with
    keypoints_xy and unpacked_correspondences_xy such that correspondence_mask[i]
    is True when a valid correspondence exists for keypoints_xy[i] and False otherwise.

    This method takes in numpy arrays and returns torch tensors. This has the effect
    of moving the data from RAM into the GPU.
    """
    unpacked_correspondences_xy = np.zeros_like(keypoints_xy)
    unpacked_correspondences_xy[:, correspondence_indices] = correspondences_xy

    correspondences_mask = np.zeros(keypoints_xy.shape[1], dtype=np.bool)
    correspondences_mask[correspondence_indices] = True

    return torch.tensor(unpacked_correspondences_xy, device="cuda"), torch.tensor(correspondences_mask, device="cuda")


def _label_inliers_outliers(img_1_keypoints_xy: torch.Tensor, img_1_correspondences_xy: torch.Tensor,
                            img_1_correspondences_mask: torch.Tensor, img_2_keypoints_xy: torch.Tensor,
                            img_2_correspondences_xy: torch.Tensor, img_2_correspondences_mask: torch.Tensor,
                            inlier_distance: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_inlier_distance = torch.tensor([inlier_distance], device="cuda")

    # img_1_inlier_distances measures how far off the keypoints predicted in image 1
    # are from the true correspondences of the keypoints predicted from image 2
    img_1_inlier_distances = torch.norm((img_1_keypoints_xy - img_2_correspondences_xy), p=2, dim=0)
    img_1_inlier_distance_mask = (img_1_inlier_distances < max_inlier_distance)

    # inliers and outliers in this anchor image are determined by the true correspondences of the keypoints
    # detected when its paired image is ran through the net as an anchor image
    img_1_inliers = img_1_inlier_distance_mask & img_2_correspondences_mask
    img_1_outliers = ~img_1_inlier_distance_mask & img_2_correspondences_mask

    img_2_inlier_distances = torch.norm((img_2_keypoints_xy - img_1_correspondences_xy), p=2, dim=0)
    img_2_inlier_distance_mask = img_2_inlier_distances < max_inlier_distance

    img_2_inliers = img_2_inlier_distance_mask & img_1_correspondences_mask
    img_2_outliers = ~img_2_inlier_distance_mask & img_1_correspondences_mask

    return img_1_inliers, img_1_outliers, img_2_inliers, img_2_outliers


def _image_to_patch_batch(image_np: np.ndarray, keypoints_xy: torch.Tensor, diameter: int) -> torch.Tensor:
    """
    Extracts keypoints_xy.shape[1] patches from image_np centered on the given keypoints.
    Returns a tensor of diameter x diameter patches in a BxCxHxW tensor. The input image is
    expected to be a numpy image in either HxW or HxWxC format.
    """
    if diameter % 2 != 1:
        raise ValueError("diameter must be odd")

    keypoints_xy = keypoints_xy.to(torch.int)

    # this will copy the image from RAM to to the GPU,
    image = epipolar_nn.dataloaders.image.load_image_for_torch(image_np)
    radius = (diameter - 1) // 2

    batch = torch.zeros((keypoints_xy.shape[1], image.shape[0], diameter, diameter), device="cuda")
    for point_idx in range(keypoints_xy.shape[1]):
        keypoint_x = keypoints_xy[0, point_idx]
        keypoint_y = keypoints_xy[1, point_idx]
        batch[point_idx, :, :, :] = image[
                                    :,
                                    keypoint_y - radius: keypoint_y + radius + 1,
                                    keypoint_x - radius: keypoint_x + radius + 1
                                    ]
    return batch
