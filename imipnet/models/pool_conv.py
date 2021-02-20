import torch
import torch.nn.functional

from . import imips


class PoolConv(imips.ImipNet):

    def __init__(self, extra_convolutions: int = 8, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        self._receptive_field_diameter = 1 + 4 * 2 + 1 * 4 + 1 * 2 + extra_convolutions * 4

        num_channels_first_half = output_channels // 2

        layers_list = []
        layers_list.extend([
            # Use 5x5 derivative to to allow for 2nd order derivatives, considered as 2 conv layers
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=5, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),  # default for tensorflow/ imips
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
        ])  # Reduces 8 px (4 from each side)

        for i in range(extra_convolutions // 2):
            layers_list.extend([
                torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.02),
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
            ])  # Reduces 4px (2 from each side)

        layers_list.extend([
            torch.nn.Conv2d(num_channels_first_half, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        ])  # Reduces 4px (2 from each side)

        for i in range(extra_convolutions // 2):
            layers_list.extend([
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
                torch.nn.LeakyReLU(negative_slope=0.02),
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            ])  # Reduces 4px (2 from each side)

        layers_list.extend([
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
        ])  # Reduces 2px (1 from each side)

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
            images = torch.nn.functional.pad(images, [self.receptive_field_radius(), self.receptive_field_radius(),
                                                      self.receptive_field_radius(), self.receptive_field_radius(),
                                                      0, 0, 0, 0], value=float('-inf'))

        return images

    def receptive_field_radius(self) -> int:
        return (self._receptive_field_diameter - 1) // 2

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter
