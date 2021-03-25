import torch
import torch.nn.functional

from . import imips


class PoolConv(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True, pool_kernel_size: int = 3):
        imips.ImipNet.__init__(self, input_channels, output_channels)
        num_pools = 4
        num_convolutions = 14 - num_pools
        pool_pad = (pool_kernel_size - 1) // 2
        self._pad = num_convolutions * 1 + num_pools * pool_pad
        self._receptive_field_diameter = 1 + 2 * self._pad

        num_channels_first_half = output_channels // 2

        layers_list = []
        layers_list.extend([
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),  # default for tensorflow/ imips,
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(num_channels_first_half, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
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
            images = torch.nn.functional.pad(images, [self._pad, self._pad,
                                                      self._pad, self._pad,
                                                      0, 0, 0, 0], value=float('-inf'))

        return images

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter


class PoolConvLowChan(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 32,
                 bias: bool = True, pool_kernel_size: int = 7):
        imips.ImipNet.__init__(self, input_channels, output_channels)
        num_pools = 4
        num_convolutions = 14 - num_pools
        pool_pad = (pool_kernel_size - 1) // 2
        self._pad = num_convolutions * 1 + num_pools * pool_pad
        self._receptive_field_diameter = 1 + 2 * self._pad

        num_channels_first_half = output_channels

        layers_list = []
        layers_list.extend([
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),  # default for tensorflow/ imips,
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(num_channels_first_half, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=0),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.02),
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
            images = torch.nn.functional.pad(images, [self._pad, self._pad,
                                                      self._pad, self._pad,
                                                      0, 0, 0, 0], value=float('-inf'))

        return images

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter
