import math

import torch

from . import hessian
from . import imips


class SqueezeExcite(torch.nn.Module):
    def __init__(self, input_channels: int, squeeze_bottleneck_factor: float):
        super(SqueezeExcite, self).__init__()
        squeeze_channels = int(squeeze_bottleneck_factor * input_channels + 0.5)
        self.squeeze = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(input_channels, squeeze_channels, kernel_size=1),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(squeeze_channels, input_channels, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.squeeze(x)


class MBConv(imips.ImipNet):
    def __init__(self, input_channels: int = 1, output_channels: int = 128):
        super(MBConv, self).__init__(input_channels, output_channels)
        """
        The layers 
        MBConv.InverseBottleneck(32, 16, inverse_bottleneck_factor=1),
        MBConv.InverseBottleneck(16, 24),
        MBConv.InverseBottleneck(24, 24),
        MBConv.InverseBottleneck(24, 40),

        Were replaced with:
        MBConv.InverseBottleneck(32, 40),

        Reducing the feature map seemed to generate a lot of noise in the 
        output results.

        Removed batch norm and drop out. Changed swish to leaky relu.
        Completely changed channel pattern to slowly expand as in SC 14. 
        
        Still not working. Plan start from the other way back out.
        Make it as similar to SC 14 as possible (squeeze excite/ skips/ channel bottlenecks)
        """
        self._receptive_field_diameter = 2 * 14 + 1

        min_pyramid_size = 5
        self.hessian_module = hessian.MaxScaleHessian(
            max_octaves=math.floor(math.log2(self._receptive_field_diameter / min_pyramid_size)) + 1,
            scales_per_octave=3,
        )

        # Add the hessian response channels to the input channels
        input_channels *= 2

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 1 * output_channels // 4, kernel_size=3),
            torch.nn.LeakyReLU(negative_slope=0.02),
            MBConv.InverseBottleneck(1 * output_channels // 4, 1 * output_channels // 4),
            MBConv.InverseBottleneck(1 * output_channels // 4, 1 * output_channels // 4),
            MBConv.InverseBottleneck(1 * output_channels // 4, 2 * output_channels // 4),
            MBConv.InverseBottleneck(2 * output_channels // 4, 2 * output_channels // 4),
            MBConv.InverseBottleneck(2 * output_channels // 4, 2 * output_channels // 4),
            MBConv.InverseBottleneck(2 * output_channels // 4, 2 * output_channels // 4),
            MBConv.InverseBottleneck(2 * output_channels // 4, 3 * output_channels // 4),
            MBConv.InverseBottleneck(3 * output_channels // 4, 3 * output_channels // 4),
            MBConv.InverseBottleneck(3 * output_channels // 4, 3 * output_channels // 4),
            MBConv.InverseBottleneck(3 * output_channels // 4, 3 * output_channels // 4),
            MBConv.InverseBottleneck(3 * output_channels // 4, 4 * output_channels // 4),
            MBConv.InverseBottleneck(4 * output_channels // 4, 4 * output_channels // 4),
            MBConv.InverseBottleneck(4 * output_channels // 4, 4 * output_channels // 4),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=1)
        )

        def init_weights(module: torch.nn.Module):
            if type(module) in [torch.nn.Conv2d]:
                torch.nn.init.kaiming_uniform_(module.weight, a=0.02, nonlinearity='leaky_relu')
                torch.nn.init.constant_(module.bias, 0)

        self.network.apply(init_weights)

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        # imips centers the data between [-127, 128]
        images = images - 127
        # scale the hessian response to [-127, 128]
        max_scale_hessian = self.hessian_module(images)
        max_scale_hessian = (max_scale_hessian * 255 / max_scale_hessian.max()) - 127

        images = self.network(torch.cat([images, max_scale_hessian], dim=1))
        if keepDim:
            images = torch.nn.functional.pad(images, [14, 14, 14, 14], value=float('-inf'))
        return images

    # Default inverse_botleneck_factor is 6 in efficient net
    # Out of memory with factor set to 6 on KITTI single gray image ... trying 2

    class InverseBottleneck(torch.nn.Module):
        def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3,
                     inverse_bottleneck_factor: float = 2, squeeze_bottleneck_factor: float = 0.25,
                     include_skip: bool = True):
            super(MBConv.InverseBottleneck, self).__init__()
            inverse_bottleneck_channels = int(inverse_bottleneck_factor * input_channels + 0.5)
            self.inverse_bottleneck_conv = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, inverse_bottleneck_channels, kernel_size=1),
                torch.nn.LeakyReLU(negative_slope=0.02),
            )
            self.depthwise_conv = torch.nn.Sequential(
                torch.nn.Conv2d(inverse_bottleneck_channels, inverse_bottleneck_channels,
                                groups=inverse_bottleneck_channels, kernel_size=kernel_size),
                torch.nn.LeakyReLU(negative_slope=0.02),
            )
            self.squeeze_excite = SqueezeExcite(inverse_bottleneck_channels, squeeze_bottleneck_factor)
            self.project_conv = torch.nn.Sequential(
                torch.nn.Conv2d(inverse_bottleneck_channels, output_channels, kernel_size=1),
                torch.nn.LeakyReLU(negative_slope=0.02),
            )
            self.sso = (kernel_size - 1) // 2  # spatial shrink offset
            self.include_skip = include_skip and input_channels == output_channels

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            work_data = self.inverse_bottleneck_conv(inputs)
            work_data = self.depthwise_conv(work_data)  # removes (kernel_size - 1) / 2 from each side
            work_data = self.squeeze_excite(work_data)
            work_data = self.project_conv(work_data)
            if self.include_skip:
                work_data = work_data + inputs[:, :, self.sso:-self.sso, self.sso:-self.sso]
            return work_data
