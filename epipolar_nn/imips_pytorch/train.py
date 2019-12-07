from typing import Iterator

import torch.utils.data

import epipolar_nn.dataloaders.pair
import epipolar_nn.dataloaders.tum
import epipolar_nn.imips_pytorch.networks.imips
from .patches import generate_training_patches


def main():
    data_root = "./data"
    iterations = 100000  # default in imips. TUM dataset has 2336234 pairs ...
    tum_dataset = epipolar_nn.dataloaders.tum.TUMMonocularStereoPairs(root=data_root, train=True, download=True)
    loader = torch.utils.data.DataLoader(tum_dataset, batch_size=1, shuffle=True)
    print(len(tum_dataset))


class ImipsTrainer:

    def __init__(self, network: epipolar_nn.imips_pytorch.networks.imips.ImipsNet, dataset: torch.utils.data.Dataset):
        self._network = network
        self._dataset = dataset
        self._dataset_iter = self._create_new_load_iterator()

    def _create_new_load_iterator(self) -> Iterator[epipolar_nn.dataloaders.pair.StereoPair]:
        return iter(torch.utils.data.DataLoader(self._dataset, batch_size=1, shuffle=True))

    def _next_pair(self) -> epipolar_nn.dataloaders.pair.StereoPair:
        try:
            return next(self._dataset_iter)
        except StopIteration:
            self._dataset_iter = self._create_new_load_iterator()
            return next(self._dataset_iter)

    def _train_patches(self, anchor_images: torch.Tensor, correspondence_images: torch.Tensor,
                       inlier_labels: torch.Tensor,
                       outlier_labels: torch.Tensor):

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

        # comments assume anchor_images and correspondence_images are both pulled from image 1 in a pair
        assert len(anchor_images.shape) == len(correspondence_images.shape) == 4 and \
               anchor_images.shape[3] == correspondence_images.shape[3] == self._network.input_channels()

        assert len(inlier_labels.shape) == len(outlier_labels.shape) == 1

        assert inlier_labels.shape[0] == \
               outlier_labels.shape[0] == anchor_images.shape[0] == correspondence_images.shape[0]

        assert self._network.receptive_field_diameter() == anchor_images.shape[1] == anchor_images.shape[2] == \
               correspondence_images.shape[1] == correspondence_images.shape[2]

        # Bx1x1xC where B == C
        anchor_outputs: torch.Tensor = self._network(anchor_images, False)
        correspondence_outputs: torch.Tensor = self._network(correspondence_images, False)

        assert anchor_outputs.shape[0] == anchor_outputs.shape[3] == correspondence_outputs.shape[0] == \
               correspondence_outputs.shape[3]

        assert anchor_outputs.shape[1] == anchor_outputs.shape[2] == correspondence_outputs.shape[1] == \
               correspondence_outputs.shape[2] == 1

        # Get the aligned, outlier responses where i \in B == j \in C
        # The following two losses work together to push down errant predictions
        # and promote responses to the correct predictions
        aligned_outlier_index = torch.diag(outlier_labels).unsqueeze(1).unsqueeze(1)

        # Grab the scores at the true correspondences sites in image 1 for each channel
        # which did not align it's anchor prediction in image 1 with the
        # true correspondence in image 1 from it's anchor predictions in image 2
        aligned_corr_outlier_scores = correspondence_outputs[aligned_outlier_index]
        assert len(aligned_corr_outlier_scores.shape) == 1

        # lower scores at the true correspondence sites lead to higher losses
        # This will promote the channel's response to the true correspondence in
        # image 1 from it's anchor prediction in image 2
        # this is called correspondence loss by imips
        outlier_correspondence_loss = torch.sum(-1 * torch.log(aligned_corr_outlier_scores))

        # Grab the scores at the predicted anchor sites in image 1 for each channel
        # which did not align it's anchor prediction in image 1 with the
        # true correspondence in image 1 from it's anchor predictions in image 2
        aligned_anchor_outlier_scores = anchor_outputs[aligned_outlier_index]

        # higher scores at the mispredicted anchor sites lead to higher losses
        # This will reduce the channel's response to the mispredicted anchor
        # in image 1
        # this is called outlier loss by imips
        outlier_anchor_loss = torch.sum(-1 * torch.log(-1 * aligned_anchor_outlier_scores + 1))

        outlier_loss = outlier_correspondence_loss + outlier_anchor_loss

        # Get the aligned, inlier responses where i \in B == j \in C
        aligned_inlier_index = torch.diag(inlier_labels).unsqueeze(1).unsqueeze(1)
        # Grab the scores at the anchor sites in image 1 for each channel
        # which did align it's anchor prediction in image 1 with the true
        # correspondence in image 1 from it's anchor prediction in image 2
        aligned_anchor_inlier_scores = anchor_outputs[aligned_inlier_index]
        assert len(aligned_anchor_inlier_scores.shape) == 1

        # lower scores at the anchor sites lead to higher losses
        # This will reinforce the channel's response at the correct anchor prediction
        # in image 1
        inlier_loss = torch.sum(-1 * torch.log(aligned_anchor_inlier_scores))

        # Next, we build a suppression loss push down any responses fromu naligned channels
        # on images for which their aligned channel produced an inlier prediction

        # We can think about the broadcasted xor broadcasting across the columns of
        # an image x channel matrix
        suppression_index = aligned_inlier_index ^ inlier_labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        unaligned_scores = anchor_outputs[suppression_index]

        # imips just adds the unaligned scores to the loss directly
        suppression_loss = torch.sum(unaligned_scores)

        loss = outlier_loss + inlier_loss + sum(unaligned_scores)

        # TODO: something with the loss

    def _train_pair(self, pair: epipolar_nn.dataloaders.pair.StereoPair):
        image_1 = torch.tensor(pair.image_1, requires_grad=False)
        image_2 = torch.tensor(pair.image_2, requires_grad=False)
        image_1_keypoints = self._network.extract_keypoints(image_1)
        image_2_keypoints = self._network.extract_keypoints(image_2)

        image_1_anchor_patches, image_1_corr_patches, image_1_inlier_labels, image_1_outlier_labels, \
        image_2_anchor_patches, image_2_corr_patches, image_2_inlier_labels, image_2_outlier_labels = \
            generate_training_patches(
                pair, image_1_keypoints,
                image_2_keypoints,
                self._network.receptive_field_diameter(),
                inlier_distance=3
            )

        self._train_patches(image_1_anchor_patches, image_1_corr_patches, image_1_inlier_labels, image_1_outlier_labels)
        self._train_patches(image_2_anchor_patches, image_2_corr_patches, image_2_inlier_labels, image_2_outlier_labels)

    def train(self, iterations: int):
        for iteration in range(1, iterations + 1):
            curr_pair = self._next_pair()
            self._train_pair(curr_pair)

            # send each image through net


if __name__ == "__main__":
    main()
