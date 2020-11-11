from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import torch


class ImipLoss(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(ImipLoss, self).__init__()

    @property
    def needs_correspondence_outputs(self) -> bool:
        return True

    @abstractmethod
    def forward_with_log_data(self, maximizer_outputs: torch.Tensor, correspondence_outputs: torch.Tensor,
                              inlier_labels: torch.Tensor, outlier_labels: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def forward(self, maximizer_outputs: torch.Tensor, correspondence_outputs: torch.Tensor,
                inlier_labels: torch.Tensor, outlier_labels: torch.Tensor) -> torch.Tensor:
        return self.forward_with_log_data(maximizer_outputs, correspondence_outputs, inlier_labels, outlier_labels)[0]
