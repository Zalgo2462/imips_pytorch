import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional

from . import convnet
from . import hessian
from . import imips


class HessConv(imips.ImipNet):

    def __init__(self, num_convolutions: int, input_channels: int, output_channels: int, bias: bool = True,
                 simple_conv_constructor: Optional[Callable[[int, int, int, Optional[bool]], imips.ImipNet]] = None):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        if simple_conv_constructor is None:
            simple_conv_constructor = convnet.SimpleConv

        # Add the hessian response channels to the input channels
        input_channels *= 2
        self._simple_conv = simple_conv_constructor(num_convolutions, input_channels, output_channels, bias=bias)

        min_pyramid_size = 5
        self.hessian_module = hessian.MaxScaleHessian(
            max_octaves=math.floor(math.log2(self._simple_conv.receptive_field_diameter() / min_pyramid_size)) + 1,
            scales_per_octave=3,
        )

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        max_scale_hessian = self.hessian_module(images)
        max_scale_hessian = max_scale_hessian * 255 / max_scale_hessian.max()  # scale the hessian response to [0, 255]

        output = self._simple_conv(torch.cat([images, max_scale_hessian], dim=1), keepDim=keepDim)
        return output

    def receptive_field_diameter(self) -> int:
        return self._simple_conv.receptive_field_diameter()

    def regularizer(self) -> Union[torch.Tensor, int, float]:
        return self._simple_conv.regularizer()


class HessBNInputConv(imips.ImipNet):

    def __init__(self, num_convolutions: int, input_channels: int, output_channels: int, bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        simple_conv_constructor = convnet.SimpleConvNoNorm

        self.input_batch_norm = torch.nn.BatchNorm2d(input_channels)

        # Add the hessian response channels to the input channels
        input_channels *= 2
        self._simple_conv = simple_conv_constructor(num_convolutions, input_channels, output_channels, bias=bias)

        min_pyramid_size = 5
        self.hessian_module = hessian.MaxScaleHessian(
            max_octaves=math.floor(math.log2(self._simple_conv.receptive_field_diameter() / min_pyramid_size)) + 1,
            scales_per_octave=3,
        )

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        images = self.input_batch_norm(images)
        max_scale_hessian = self.hessian_module(images)
        output = self._simple_conv(torch.cat([images, max_scale_hessian], dim=1), keepDim=keepDim)
        return output

    def receptive_field_diameter(self) -> int:
        return self._simple_conv.receptive_field_diameter()

    def regularizer(self) -> Union[torch.Tensor, int, float]:
        return self._simple_conv.regularizer()
