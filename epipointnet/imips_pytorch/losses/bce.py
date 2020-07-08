from typing import Tuple, Dict

import torch

from .imips import ImipLoss


class BCELoss(ImipLoss):

    def __init__(self, bce_pos_weight: float = 1, epsilon: float = 1e-4):
        super(BCELoss, self).__init__()
        self._epsilon = torch.nn.Parameter(torch.tensor([epsilon]), requires_grad=False)
        self._bce_module = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]), reduction="sum")

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
            # correspondence_outputs = correspondence_outputs[:, :, center_px, center_px]

        maximizer_outputs = maximizer_outputs.squeeze()  # BxCx1x1 -> BxC

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

        # convert inlier/ outlier labels to a single array for valid data
        has_data_labels = inlier_labels | outlier_labels

        # create indexes into maximizer outputs
        aligned_data_index = torch.diag(has_data_labels)
        # equivalent: inlier_labels.unsqueeze(1).repeat(1, inlier_labels.shape[1]) - inlier_labels.diag()
        unaligned_inlier_index = torch.diag(inlier_labels) ^ inlier_labels.unsqueeze(1)

        aligned_bce_outputs = maximizer_outputs[aligned_data_index]
        unaligned_bce_outputs = maximizer_outputs[unaligned_inlier_index]

        aligned_bce_labels = inlier_labels[has_data_labels].to(dtype=maximizer_outputs.dtype)
        unaligned_bce_labels = torch.zeros(unaligned_bce_outputs.numel(),
                                           dtype=maximizer_outputs.dtype, device=maximizer_outputs.device)

        num_samples = torch.tensor(aligned_bce_labels.numel() + unaligned_bce_labels.numel(),
                                   dtype=maximizer_outputs.dtype, device=maximizer_outputs.device)

        # Apply binary cross entropy to the network's outputs on the maximizing patch for each channel
        if aligned_bce_labels.numel() == 0:
            aligned_bce_loss = torch.zeros([1], requires_grad=True, device=maximizer_outputs.device)
        else:
            aligned_bce_loss = self._bce_module(aligned_bce_outputs, aligned_bce_labels) / num_samples

        # Finally, if a channel attains its maximum response inside of a given radius
        # about it's target correspondence site, the responses of all the other channels
        # to it's maximizing patch are minimized.
        if unaligned_bce_labels.numel() == 0:
            unaligned_bce_loss = torch.zeros([1], requires_grad=True, device=maximizer_outputs.device)
        else:
            unaligned_bce_loss = self._bce_module(unaligned_bce_outputs, unaligned_bce_labels) / num_samples

        loss = aligned_bce_loss + unaligned_bce_loss

        return loss, {
            "loss": loss.detach(),
            "aligned_bce_loss": aligned_bce_loss.detach(),
            "unaligned_bce_loss": unaligned_bce_loss.detach(),
        }
