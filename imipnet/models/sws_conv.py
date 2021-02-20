import torch
import torch.nn.functional
from torch import nn

from . import imips


class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, gain, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.tensor(gain), requires_grad=False)
        self.fan_in_var_correction = nn.Parameter(torch.tensor(self.weight[0].numel(), dtype=torch.float32).sqrt(),
                                                  requires_grad=False)
        self.eps = 1e-5

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
        weight = self.gain * (weight / (std.expand_as(weight) * self.fan_in_var_correction + self.eps))

        return torch.nn.functional.conv2d(x, weight, self.bias, self.stride,
                                          self.padding, self.dilation, self.groups)


class SWSConv(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        # gain = torch.nn.functional.leaky_relu(torch.normal(0, 1, (2 ** 16, 256)), 0.02).var(dim=1).mean() ** -0.5
        gain = 1.6969

        self._num_convolutions = num_convolutions
        self._receptive_field_diameter = num_convolutions * 2 + 1

        num_channels_first_half = output_channels // 2

        layers_list = []
        layers_list.extend([
            ScaledWSConv2d(gain, input_channels, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02)  # default for tensorflow/ imips
        ])

        for i in range((num_convolutions // 2) - 1):
            layers_list.extend([
                ScaledWSConv2d(gain, num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.02)
            ])

        layers_list.extend([
            ScaledWSConv2d(gain, num_channels_first_half, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02)
        ])

        for i in range((num_convolutions // 2) - 2):
            layers_list.extend([
                ScaledWSConv2d(gain, output_channels, output_channels, kernel_size=3, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.02)
            ])

        layers_list.extend([
            ScaledWSConv2d(gain, output_channels, output_channels, kernel_size=3, bias=bias),
        ])

        self.conv_layers = torch.nn.Sequential(*layers_list)

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        images = self.conv_layers(images)

        # imips subtracts off 1.5 without explanation
        images = images - 1.5

        if keepDim:
            images = torch.nn.functional.pad(images, [self._num_convolutions, self._num_convolutions,
                                                      self._num_convolutions, self._num_convolutions,
                                                      0, 0, 0, 0], value=float('-inf'))

        return images

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter
