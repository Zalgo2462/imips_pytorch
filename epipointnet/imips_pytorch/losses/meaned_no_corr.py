from typing import Tuple, Dict

import torch

from .imips import ImipLoss


class MeanedNoCorrImipLoss(ImipLoss):

    def __init__(self, epsilon: float = 1e-4):
        super(MeanedNoCorrImipLoss, self).__init__()
        self._epsilon = torch.nn.Parameter(torch.tensor([epsilon]), requires_grad=False)

    @property
    def needs_correspondence_outputs(self) -> bool:
        return False

    def forward_with_log_data(self, maximizer_outputs: torch.Tensor, correspondence_outputs: torch.Tensor,
                              inlier_labels: torch.Tensor, outlier_labels: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        # maximizer_outputs: BxCx1x1 where B == C
        # correspondence_outputs: BxCx1x1 where B == C
        # If h and w are not 1 w.r.t. maximizer_outputs and correspondence_outputs,
        # the center values will be extracted.

        assert (maximizer_outputs.shape[0] == maximizer_outputs.shape[1] ==
                correspondence_outputs.shape[0] == correspondence_outputs.shape[1])

        assert (maximizer_outputs.shape[2] == maximizer_outputs.shape[3] ==
                correspondence_outputs.shape[2] == correspondence_outputs.shape[3])

        if maximizer_outputs.shape[2] != 1:
            center_px = (maximizer_outputs.shape[2] - 1) // 2
            maximizer_outputs = maximizer_outputs[:, :, center_px, center_px]
            correspondence_outputs = correspondence_outputs[:, :, center_px, center_px]

        maximizer_outputs = torch.sigmoid(maximizer_outputs.squeeze())  # BxCx1x1 -> BxC

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

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

        # If a channel's maximum response is outside of a given radius about the target correspondence site, the
        # channel's response to it's maximizing patch in image 1 is minimized.
        aligned_outlier_maximizer_scores = maximizer_outputs[aligned_outlier_index]

        # A higher response to a channel's maximizing patch in image 1 will lead to a
        # higher loss for a channel which attains its maximum outside of a given radius
        # about it's target correspondence site. This is called outlier loss by imips.
        outlier_maximizer_loss = torch.mean(
            -1 * torch.log(torch.max(-1 * aligned_outlier_maximizer_scores + 1, self._epsilon)))
        if aligned_outlier_maximizer_scores.nelement() == 0:
            outlier_maximizer_loss = torch.zeros([1], requires_grad=True, device=outlier_maximizer_loss.device)

        # If a channel's maximum response is inside of a given radius about the target correspondence site, the
        # chanel's response to it's maximizing patch in image 1 is maximized.

        # grabs the inlier responses where the batch index and channel index align.
        aligned_inlier_index = torch.diag(inlier_labels)

        aligned_inlier_maximizer_scores = maximizer_outputs[aligned_inlier_index]

        # A lower response to a channel's maximizing patch in image 1 wil lead to
        # a higher loss for a channel which attains its maximum inside of a given radius
        # about it's target correspondence site. This is called inlier_loss by imips.
        inlier_loss = torch.mean(-1 * torch.log(aligned_inlier_maximizer_scores + self._epsilon))
        if aligned_inlier_maximizer_scores.nelement() == 0:
            inlier_loss = torch.zeros([1], requires_grad=True, device=inlier_loss.device)

        # Finally, if a channel attains its maximum response inside of a given radius
        # about it's target correspondence site, the responses of all the other channels
        # to it's maximizing patch are minimized.
        # equivalent: inlier_labels.unsqueeze(1).repeat(1, inlier_labels.shape[1]) - inlier_labels.diag()
        unaligned_inlier_index = aligned_inlier_index ^ inlier_labels.unsqueeze(1)

        # Changed the unaligned_maximizer_loss to NLL.
        # The summed version of this loss would introduce a penalty on each new inlier.
        # The meaned version of this loss over all unaligned channel responses does not.
        # Here, we first find the mean unaligned response to each inlier producing patch, and then sum
        # over all of the meaned responses.

        # Mask out the diagonal and the rows where we don't have inliers
        unaligned_maximizer_loss_mat = -1 * torch.log(torch.max(-1 * maximizer_outputs + 1, self._epsilon))
        unaligned_maximizer_loss_mat[torch.logical_not(unaligned_inlier_index)] = 0
        # Get the number of elements in each row so we can average the rows
        num_unaligned_outputs = unaligned_inlier_index.sum(dim=1).clamp_min(1).unsqueeze(1)
        unaligned_maximizer_loss = (unaligned_maximizer_loss_mat / num_unaligned_outputs).sum()

        loss = outlier_maximizer_loss + inlier_loss + unaligned_maximizer_loss

        return loss, {
            "loss": loss.detach(),
            "inlier_maximizer_loss": inlier_loss.detach(),
            "outlier_maximizer_loss": outlier_maximizer_loss.detach(),
            "unaligned_maximizer_loss": unaligned_maximizer_loss.detach(),
        }
