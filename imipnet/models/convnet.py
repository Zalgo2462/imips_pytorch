import torch
import torch.nn.functional

from . import imips


class SimpleConv(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        self._num_convolutions = num_convolutions
        self._receptive_field_diameter = num_convolutions * 2 + 1

        num_channels_first_half = output_channels // 2

        layers_list = []
        layers_list.extend([
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02)  # default for tensorflow/ imips
        ])

        for i in range((num_convolutions // 2) - 1):
            layers_list.extend([
                torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.02)
            ])

        layers_list.extend([
            torch.nn.Conv2d(num_channels_first_half, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02)
        ])

        for i in range((num_convolutions // 2) - 2):
            layers_list.extend([
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.02)
            ])

        layers_list.extend([
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
        ])

        self.conv_layers = torch.nn.Sequential(*layers_list)

        def init_weights(module: torch.nn.Module):
            if type(module) in [torch.nn.Conv2d]:
                torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.02))
                if bias:
                    torch.nn.init.constant_(module.bias, 0)

        self.conv_layers.apply(init_weights)

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
