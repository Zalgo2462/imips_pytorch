from typing import Tuple, Dict

import torch

from .imips import ImipLoss


class OHNMAvgLossNoOCL(ImipLoss):

    def __init__(self, epsilon: float = 1e-4):
        super(OHNMAvgLossNoOCL, self).__init__()
        self._epsilon = torch.nn.Parameter(torch.tensor([epsilon]), requires_grad=False)

    @property
    def needs_correspondence_outputs(self) -> bool:
        return True

    @staticmethod
    def _add_if_not_none(x, y):
        return x + y if y is not None else x

    @staticmethod
    def _detach_if_not_none(x):
        return x.detach() if x is not None else None

    def forward_with_log_data(self, maximizer_outputs: torch.Tensor, correspondence_outputs: torch.Tensor,
                              inlier_labels: torch.Tensor, outlier_labels: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # maximizer_outputs: BNxCx1x1 where B == C
        # correspondence_outputs: BxCx1x1 where B == C
        # inlier_labels: B
        # outlier_labels: B
        # If h and w are not 1 w.r.t. maximizer_outputs and correspondence_outputs,
        # the center values will be extracted.

        assert (maximizer_outputs.shape[0] % maximizer_outputs.shape[1] == 0)
        assert (maximizer_outputs.shape[2] == maximizer_outputs.shape[3])

        if maximizer_outputs.shape[2] != 1:
            center_px = (maximizer_outputs.shape[2] - 1) // 2
            maximizer_outputs = maximizer_outputs[:, :, center_px, center_px]

        maximizer_outputs = maximizer_outputs.squeeze()  # BNxCx1x1 -> BNxC

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

        # expand inlier_labels and has_data_labels by num_patches_per_channel
        # inlier should begin every segment of (num patches per channel)
        num_patches_per_channel = maximizer_outputs.shape[0] // maximizer_outputs.shape[1]

        expanded_inlier_labels = torch.zeros(
            inlier_labels.shape[0] * num_patches_per_channel,
            dtype=inlier_labels.dtype, device=inlier_labels.device
        )
        maximum_patch_index = torch.arange(0, maximizer_outputs.shape[0], num_patches_per_channel)
        expanded_inlier_labels[maximum_patch_index] = inlier_labels

        has_data_labels = inlier_labels | outlier_labels  # B
        expanded_has_data_labels = has_data_labels.repeat_interleave(num_patches_per_channel)
        expanded_outlier_labels = (
                expanded_has_data_labels & torch.logical_not(expanded_inlier_labels).to(dtype=torch.uint8)
        )

        maxima_inlier_index_2d = expanded_inlier_labels[:, None] * torch.repeat_interleave(
            torch.eye(
                maximizer_outputs.shape[1], device=expanded_inlier_labels.device, dtype=expanded_inlier_labels.dtype
            ), num_patches_per_channel, dim=0
        )

        maxima_outlier_index_2d = expanded_outlier_labels[:, None] * torch.repeat_interleave(
            torch.eye(
                maximizer_outputs.shape[1], device=expanded_outlier_labels.device, dtype=expanded_outlier_labels.dtype
            ), num_patches_per_channel, dim=0
        )

        maximizer_inlier_outputs = maximizer_outputs[maxima_inlier_index_2d]
        if maximizer_inlier_outputs.numel() == 0:
            maxima_inlier_loss = None
        else:
            maxima_inlier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                maximizer_inlier_outputs, torch.ones_like(maximizer_inlier_outputs), reduction="mean"
            )

        maximizer_outlier_outputs = maximizer_outputs[maxima_outlier_index_2d]
        if maximizer_outlier_outputs.numel() == 0:
            maxima_outlier_loss = None
        else:
            maxima_outlier_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                maximizer_outlier_outputs, torch.zeros_like(maximizer_outlier_outputs), reduction="mean"
            )

        # Finally, if a channel attains its maximum response inside of a given radius
        # about it's target correspondence site, the responses of all the other channels
        # to it's maximizing patch are minimized.

        # remove non-maxima patches from maximizer outputs
        maximizer_outputs = maximizer_outputs[maximum_patch_index]  # BxC where B==C

        aligned_inlier_index = torch.diag(inlier_labels)
        # equivalent: inlier_labels.unsqueeze(1).repeat(1, inlier_labels.shape[1]) - inlier_labels.diag()
        unaligned_inlier_index = aligned_inlier_index ^ inlier_labels.unsqueeze(1)
        # unaligned_inlier_index = unaligned_inlier_index.triu()  # break ties
        unaligned_inlier_outputs = maximizer_outputs[unaligned_inlier_index]

        if unaligned_inlier_outputs.numel() == 0:
            unaligned_maxima_losses = None
        else:
            unaligned_bce_labels = torch.zeros_like(unaligned_inlier_outputs)
            unaligned_maxima_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                unaligned_inlier_outputs, unaligned_bce_labels, reduction="sum")

        total_loss = torch.zeros(1, device=maximizer_outputs.device, dtype=maximizer_outputs.dtype, requires_grad=True)
        total_loss = self._add_if_not_none(total_loss, maxima_inlier_loss)
        total_loss = self._add_if_not_none(total_loss, maxima_outlier_loss)
        total_loss = self._add_if_not_none(total_loss, unaligned_maxima_losses)

        return total_loss, {
            "loss": total_loss.detach(),
            "inlier_maximizer_loss": self._detach_if_not_none(maxima_inlier_loss),
            "outlier_maximizer_loss": self._detach_if_not_none(maxima_outlier_loss),
            "unaligned_maximizer_loss": self._detach_if_not_none(unaligned_maxima_losses)
        }
