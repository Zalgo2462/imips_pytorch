from typing import Union

import torch

from . import imips
from .convnet import SimpleConv


class OrthoSimpleConv(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)
        self._simple_conv = SimpleConv(num_convolutions, input_channels, output_channels, bias)
        self._final_conv = torch.nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=False)
        torch.nn.init.orthogonal_(self._final_conv.weight)

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        return self._final_conv(self._simple_conv(images, keepDim=keepDim))

    def receptive_field_diameter(self) -> int:
        return self._simple_conv.receptive_field_diameter()

    def regularizer(self) -> Union[torch.Tensor, int, float]:
        output_weight_mat = self._final_conv.weight.view((self._final_conv.weight.shape[0], -1))
        return 1e-2 * torch.norm(
            output_weight_mat.mm(output_weight_mat.t()) - torch.eye(self._final_conv.weight.shape[0],
                                                                    device=output_weight_mat.device,
                                                                    dtype=output_weight_mat.dtype),
            p="fro"
        )
