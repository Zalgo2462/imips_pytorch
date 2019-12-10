import torch
import torch.nn.functional

from . import imips


class SimpleConv(imips.ImipsNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128):
        imips.ImipsNet.__init__(self, input_channels, output_channels)

        self._num_convolutions = num_convolutions
        self._receptive_field_diameter = num_convolutions * 2 + 1

        num_channels_first_half = output_channels // 2

        layers_list = []
        layers_list.append(
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=3)
        )
        layers_list.append(
            torch.nn.LeakyReLU(negative_slope=0.02)  # default for tensorflow/ imips
        )

        for i in range((num_convolutions // 2) - 1):
            layers_list.append(
                torch.nn.Conv2d(num_channels_first_half, num_channels_first_half, kernel_size=3)
            )
            layers_list.append(
                torch.nn.LeakyReLU(negative_slope=0.02)
            )

        layers_list.append(
            torch.nn.Conv2d(num_channels_first_half, output_channels, kernel_size=3),
        )
        layers_list.append(
            torch.nn.LeakyReLU(negative_slope=0.02)
        )

        for i in range((num_convolutions // 2) - 1):
            layers_list.append(
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=3),
            )
            layers_list.append(
                torch.nn.LeakyReLU(negative_slope=0.02)
            )

        self.conv_layers = torch.nn.Sequential(*layers_list)

        self.sigmoid_layer = torch.nn.Sigmoid()

    def forward(self, images: torch.Tensor, keepDim: bool) -> torch.Tensor:
        # imips centers the data between [-127, 128]
        constant_bias = torch.tensor([127], device="cuda")

        images = images - constant_bias
        images = self.conv_layers(images)

        # imips subtracts off 1.5 without explanation
        sigmoid_bias = torch.tensor([1.5], device="cuda")
        images = images - sigmoid_bias

        images: torch.Tensor = self.sigmoid_layer(images)

        if keepDim:
            images = torch.nn.functional.pad(images, [self._num_convolutions, self._num_convolutions,
                                                      self._num_convolutions, self._num_convolutions,
                                                      0, 0, 0, 0])

        return images

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter
