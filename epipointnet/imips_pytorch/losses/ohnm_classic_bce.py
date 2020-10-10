from typing import Tuple, Dict

import torch

from .imips import ImipLoss


class OHNMClassicImipBCELoss(ImipLoss):

    def __init__(self, bce_pos_weight: float = 1, epsilon: float = 1e-4):
        super(OHNMClassicImipBCELoss, self).__init__()
        self._epsilon = torch.nn.Parameter(torch.tensor([epsilon]), requires_grad=False)
        self._bce_module = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight]), reduction="sum")

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
        if correspondence_outputs.shape[2] != 1:
            center_px = (correspondence_outputs.shape[2] - 1) // 2
            correspondence_outputs = correspondence_outputs[:, :, center_px, center_px]

        maximizer_outputs = maximizer_outputs.squeeze()  # BxCx1x1 -> BxC
        correspondence_outputs = correspondence_outputs.squeeze()  # BxCx1x1 -> BxC

        # convert the label types so we can use torch.diag() on the labels
        if inlier_labels.dtype == torch.bool:
            inlier_labels = inlier_labels.to(torch.uint8)

        if outlier_labels.dtype == torch.bool:
            outlier_labels = outlier_labels.to(torch.uint8)

        # Boost responses to correspondence patches if the channel returned all outliers
        corr_outlier_index_2d = torch.diag(outlier_labels)
        aligned_outlier_corr_outputs = correspondence_outputs[corr_outlier_index_2d]
        aligned_corr_labels = torch.ones_like(aligned_outlier_corr_outputs)
        if aligned_outlier_corr_outputs.numel() == 0:
            aligned_corr_losses = None
        else:
            aligned_corr_losses = self._bce_module(aligned_outlier_corr_outputs, aligned_corr_labels)

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
        maxima_has_data_index_2d = torch.repeat_interleave(
            has_data_labels[:, None] * torch.eye(
                maximizer_outputs.shape[1], device=has_data_labels.device, dtype=torch.uint8
            ),
            num_patches_per_channel, dim=0
        )  # BNxC

        expanded_has_data_labels = has_data_labels.repeat_interleave(num_patches_per_channel)

        aligned_maxima_bce_labels = expanded_inlier_labels[expanded_has_data_labels].to(dtype=maximizer_outputs.dtype)
        if aligned_maxima_bce_labels.numel() == 0:
            aligned_maxima_losses = None
        else:
            aligned_maxima_losses = self._bce_module(
                maximizer_outputs[maxima_has_data_index_2d], aligned_maxima_bce_labels
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
            unaligned_maxima_losses = torch.sigmoid(unaligned_inlier_outputs).sum()

        total_loss = torch.zeros(1, device=maximizer_outputs.device, dtype=maximizer_outputs.dtype, requires_grad=True)

        total_loss = self._add_if_not_none(total_loss, aligned_maxima_losses)
        total_loss = self._add_if_not_none(total_loss, aligned_corr_losses)
        total_loss = self._add_if_not_none(total_loss, unaligned_maxima_losses)

        return total_loss, {
            "loss": total_loss.detach(),
            "bce_maximizer_loss": self._detach_if_not_none(aligned_maxima_losses),
            "outlier_correspondence_loss": self._detach_if_not_none(aligned_corr_losses),
            "unaligned_maximizer_loss": self._detach_if_not_none(unaligned_maxima_losses)
        }
