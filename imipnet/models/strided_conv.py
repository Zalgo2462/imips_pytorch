import torch

from imipnet.models import imips


class StridedConv(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        self._num_convolutions = num_convolutions
        self._receptive_field_diameter = 2 * (((num_convolutions - 1) * 2 + 1) - 1) + 7
        self._receptive_field_radius = self._receptive_field_diameter // 2
        num_channels_first_half = output_channels // 2

        layers_list = []

        layers_list.extend([
            # Fwd size rule: 7 -> 1, 9 -> 2, 11 -> 3, 13 -> 4: 1 + ceil((x - 7) / 2)
            # Rev size rule: 2*(y-1)+7
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=7, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.02)
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
        self.__keep_dim_conv_weight = torch.nn.Parameter(
            torch.nn.functional.pad(
                torch.ones(output_channels, 1, 1, 1),
                [1, 1, 1, 1]
            ),
            requires_grad=False
        )

        def init_weights(module: torch.nn.Module):
            if type(module) in [torch.nn.Conv2d]:
                torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.02))
                if bias:
                    torch.nn.init.constant_(module.bias, 0)

        self.conv_layers.apply(init_weights)

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        in_shape = images.shape

        # imips centers the data between [-127, 128]
        images = images - 127
        images = self.conv_layers(images)

        # imips subtracts off 1.5 without explanation
        images = images - 1.5

        if keepDim:
            images = torch.nn.functional.pad(images, [
                self._num_convolutions - 1,
                self._num_convolutions - 1,
                self._num_convolutions - 1,
                self._num_convolutions - 1,
                0, 0, 0, 0], value=float('-inf'))
            images = torch.nn.functional.conv_transpose2d(images, self.__keep_dim_conv_weight, stride=2,
                                                          padding=1, groups=self.__keep_dim_conv_weight.shape[0])
            images[:, :, torch.arange(1, images.shape[2], 2), :] = float('-inf')
            images[:, :, :, torch.arange(1, images.shape[3], 2)] = float('-inf')
            images = torch.nn.functional.pad(images, [
                3, in_shape[-1] - (images.shape[-1] + 3),
                3, in_shape[-2] - (images.shape[-2] + 3),
                0, 0, 0, 0], value=float('-inf'))

        return images

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter


class StridedConvNoNorm(imips.ImipNet):

    def __init__(self, num_convolutions: int = 14, input_channels: int = 1, output_channels: int = 128,
                 bias: bool = True):
        imips.ImipNet.__init__(self, input_channels, output_channels)

        self._num_convolutions = num_convolutions
        self._receptive_field_diameter = 2 * (((num_convolutions - 1) * 2 + 1) - 1) + 7
        self._receptive_field_radius = self._receptive_field_diameter // 2
        num_channels_first_half = output_channels // 2

        layers_list = []

        layers_list.extend([
            # Fwd size rule: 7 -> 1, 9 -> 2, 11 -> 3, 13 -> 4: 1 + ceil((x - 7) / 2)
            # Rev size rule: 2*(y-1)+7
            torch.nn.Conv2d(input_channels, num_channels_first_half, kernel_size=7, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.02)
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
        self.__keep_dim_conv_weight = torch.nn.Parameter(
            torch.nn.functional.pad(
                torch.ones(output_channels, 1, 1, 1),
                [1, 1, 1, 1]
            ),
            requires_grad=False
        )

        def init_weights(module: torch.nn.Module):
            if type(module) in [torch.nn.Conv2d]:
                torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.02))
                if bias:
                    torch.nn.init.constant_(module.bias, 0)

        self.conv_layers.apply(init_weights)

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        in_shape = images.shape
        images = self.conv_layers(images)

        if keepDim:
            images = torch.nn.functional.pad(images, [
                self._num_convolutions - 1,
                self._num_convolutions - 1,
                self._num_convolutions - 1,
                self._num_convolutions - 1,
                0, 0, 0, 0], value=float('-inf'))
            images = torch.nn.functional.conv_transpose2d(images, self.__keep_dim_conv_weight, stride=2,
                                                          padding=1, groups=self.__keep_dim_conv_weight.shape[0])
            images[:, :, torch.arange(1, images.shape[2], 2), :] = float('-inf')
            images[:, :, :, torch.arange(1, images.shape[3], 2)] = float('-inf')
            images = torch.nn.functional.pad(images, [
                3, in_shape[-1] - (images.shape[-1] + 3),
                3, in_shape[-2] - (images.shape[-2] + 3),
                0, 0, 0, 0], value=float('-inf'))

        return images

    def receptive_field_diameter(self) -> int:
        return self._receptive_field_diameter
