from typing import Tuple, Dict

import torch

from .imips import ImipLoss


class BCELoss(ImipLoss):

    def __init__(self, bce_pos_weight: float = 1, epsilon: float = 1e-4):
        super(BCELoss, self).__init__()
        self._epsilon = torch.nn.Parameter(torch.tensor([epsilon]), requires_grad=False)
        self._bce_module = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]), reduction="mean")

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
            # correspondence_outputs = correspondence_outputs[:, :, center_px, center_px]

        maximizer_outputs = maximizer_outputs.squeeze()  # BxCx1x1 -> BxC

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

        has_data_labels = inlier_labels | outlier_labels
        aligned_data_index = torch.diag(has_data_labels)

        bce_labels = inlier_labels[has_data_labels].to(dtype=maximizer_outputs.dtype)
        if bce_labels.numel() == 0:
            bce_loss = torch.zeros([1], requires_grad=True, device=maximizer_outputs.device)
        else:
            bce_loss = self._bce_module(maximizer_outputs[aligned_data_index], bce_labels)

        # Finally, if a channel attains its maximum response inside of a given radius
        # about it's target correspondence site, the responses of all the other channels
        # to it's maximizing patch are minimized.

        aligned_inlier_index = torch.diag(inlier_labels)
        # equivalent: inlier_labels.unsqueeze(1).repeat(1, inlier_labels.shape[1]) - inlier_labels.diag()
        unaligned_inlier_index = aligned_inlier_index ^ inlier_labels.unsqueeze(1)

        # Changed the unaligned_maximizer_loss to NLL.
        # The summed version of this loss would introduce a penalty on each new inlier.
        # The meaned version of this loss over all unaligned channel responses does not.
        # Here, we first find the mean unaligned response to each inlier producing patch, and then sum
        # over all of the meaned responses.

        # Mask out the diagonal and the rows where we don't have inliers
        unaligned_maximizer_loss_mat = -1 * torch.log(
            torch.max(
                -1 * torch.sigmoid(maximizer_outputs) + 1,
                self._epsilon
            )
        )
        unaligned_maximizer_loss_mat[torch.logical_not(unaligned_inlier_index)] = 0
        # Get the number of elements in each row so we can average the rows
        num_unaligned_outputs = unaligned_inlier_index.sum(dim=1).clamp_min(1).unsqueeze(1)
        unaligned_maximizer_loss = (unaligned_maximizer_loss_mat / num_unaligned_outputs).sum()

        loss = bce_loss + unaligned_maximizer_loss

        return loss, {
            "loss": loss.detach(),
            "bce_loss": bce_loss.detach(),
            "unaligned_maximizer_loss": unaligned_maximizer_loss.detach(),
        }
