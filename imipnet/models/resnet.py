import itertools
from typing import Union, Tuple

import torch
from torchvision.models import ResNet as TorchResNet
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock as TorchBasicBlock

from imipnet.models import imips


class ResNet(imips.ImipNet):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(out_channels, affine=True)
            self.relu = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(out_channels, affine=True)
            if in_channels == out_channels:
                self.upsample = lambda x: x
            else:
                self.upsample = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(out_channels, affine=True)
                )

        def forward(self, in_value: Union[Tuple[torch.Tensor, bool], torch.Tensor]):
            if isinstance(in_value, tuple):
                x, keepDim = in_value
            else:
                x = in_value
                keepDim = False

            identity = self.upsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if keepDim:
                out = torch.nn.functional.pad(out, [2, 2, 2, 2], value=float('-inf'))
            else:
                identity = identity[:, :, 2:identity.shape[2] - 2, 2:identity.shape[3] - 2]
            out += identity
            out = self.relu(out)

            return out, keepDim

    def __init__(self, bunk_num_convs, in_channels=1, out_channels=128):
        super().__init__(in_channels, out_channels)
        self.blocks_per_layers = (1, 2, 2, 1)
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, padding=0,
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, affine=True)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.layer1 = self._make_layer(64, 64, self.blocks_per_layers[0])
        self.layer2 = self._make_layer(64, 128, self.blocks_per_layers[1])
        self.layer3 = self._make_layer(128, 256, self.blocks_per_layers[2])
        self.layer4 = self._make_layer(256, out_channels, self.blocks_per_layers[3])

        def init_weights(module: torch.nn.Module):
            if type(module) in [torch.nn.Conv2d]:
                torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.02))

        self.apply(init_weights)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = [self.Block(in_channels, out_channels)]
        for i in range(1, blocks):
            layers.append(self.Block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x, keepDim=False) -> torch.Tensor:
        x = self.conv1(x)
        if keepDim:
            x = torch.nn.functional.pad(x, [3, 3, 3, 3], value=float('-inf'))

        x = self.bn1(x)
        x = self.relu(x)
        x, _ = self.layer1((x, keepDim))
        x, _ = self.layer2((x, keepDim))
        x, _ = self.layer3((x, keepDim))
        x, _ = self.layer4((x, keepDim))
        return x

    def receptive_field_diameter(self) -> int:
        return sum([4 * i for i in self.blocks_per_layers]) + 6 + 1


class PatchResNet18(imips.ImipNet):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, dilate_conv_weights: Tuple[bool, bool] = None):
            super().__init__()
            self.dilate_conv_weights = dilate_conv_weights
            if self.dilate_conv_weights is None:
                self.dilate_conv_weights = (False, False)
            self.conv1_ksize = 5 if self.dilate_conv_weights[0] else 3
            self.conv2_ksize = 5 if self.dilate_conv_weights[1] else 3

            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=self.conv1_ksize, padding=0, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=self.conv2_ksize, padding=0,
                                         bias=False)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            if in_channels == out_channels:
                self.downsample = lambda x: x
            else:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(out_channels)
                )

            self.pad = (self.conv1_ksize - 1) // 2 + (self.conv2_ksize - 1) // 2

        def forward(self, in_value: Union[Tuple[torch.Tensor, bool], torch.Tensor]):
            if isinstance(in_value, tuple):
                x, keepDim = in_value
            else:
                x = in_value
                keepDim = False

            identity = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if keepDim:
                out = torch.nn.functional.pad(out, [self.pad] * 4, value=float('-inf'))
            else:
                identity = identity[:, :, self.pad:(identity.shape[2] - self.pad),
                           self.pad:(identity.shape[3] - self.pad)]
            out += identity
            out = self.relu(out)

            return out, keepDim

    def __init__(self, output_channels: int, pretrained: bool = False):
        super().__init__(3, output_channels)
        self.blocks_per_layers = (2, 2, 2, 2)
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, bias=False)
        # note dilate next layer after conv1 once since conv1 was stride = 2
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        # note conv1's dilation is folded into max pool
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1)
        # note dilate next layer after maxpool since maxpool was stride = 2
        self.layer1 = self._make_layer(64, 64, self.blocks_per_layers[0], dilate_first_conv=True)
        self.layer2 = self._make_layer(64, 128, self.blocks_per_layers[1], dilate_second_conv=True)
        self.layer3 = self._make_layer(128, 256, self.blocks_per_layers[2], dilate_second_conv=True)
        self.layer4 = self._make_layer(256, 512, self.blocks_per_layers[3], dilate_second_conv=True)

        self.freeze_backbone()

        self.attention_net = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.LeakyReLU(negative_slope=0.02, inplace=True),
            torch.nn.Conv2d(512, output_channels, kernel_size=1),
        )

        if pretrained:
            self.load_from_pytorch_resnet(resnet18(pretrained=True))

    def freeze_backbone(self):
        frozen_params = itertools.chain(
            self.conv1.parameters(True),
            self.bn1.parameters(True),
            self.layer1.parameters(True),
            self.layer2.parameters(True),
            self.layer3.parameters(True),
            self.layer4.parameters(True)
        )

        for param in frozen_params:
            param.requires_grad = False

    def load_from_pytorch_resnet(self, other: TorchResNet):
        self.conv1.load_state_dict(other.conv1.state_dict())
        self.bn1.load_state_dict(other.bn1.state_dict())

        layers_zipped = zip(
            [self.layer1, self.layer2, self.layer3, self.layer4],
            [other.layer1, other.layer2, other.layer3, other.layer4]
        )
        for (self_layer, other_layer) in layers_zipped:
            assert len(self_layer) == len(other_layer)
            for i in range(len(self_layer)):
                self_block = self_layer[i]  # type: PatchResNet18.Block
                other_block = other_layer[i]  # type: TorchBasicBlock

                self_block.bn1.load_state_dict(other_block.bn1.state_dict())
                self_block.bn2.load_state_dict(other_block.bn2.state_dict())

                if isinstance(self_block.downsample, torch.nn.Sequential):
                    self_block.downsample.load_state_dict(other_block.downsample.state_dict())

                if self_block.conv1_ksize == 5:
                    self_block.conv1.weight.data = self.dilate_weight(other_block.conv1.weight)
                else:
                    self_block.conv1.weight.data = other_block.conv1.weight

                if self_block.conv2_ksize == 5:
                    self_block.conv2.weight.data = self.dilate_weight(other_block.conv2.weight)
                else:
                    self_block.conv2.weight.data = other_block.conv2.weight

    @staticmethod
    def dilate_weight(weight: torch.Tensor) -> torch.Tensor:
        dilation_weight = torch.nn.functional.pad(
            torch.ones(1, 1, 1, 1, device=weight.device),
            [1, 1, 1, 1]
        ).expand(weight.shape[1], 1, 3, 3)
        return torch.nn.functional.conv_transpose2d(
            weight, dilation_weight, stride=2, groups=weight.shape[1])[:, :, 1:-1, 1:-1]

    def _make_layer(self, in_channels, out_channels, blocks, dilate_first_conv=False, dilate_second_conv=False):
        layers = [self.Block(in_channels, out_channels, (dilate_first_conv, dilate_second_conv))]
        for i in range(1, blocks):
            layers.append(self.Block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def receptive_field_diameter(self) -> int:
        return 47

    def forward(self, x, keepDim=False) -> torch.Tensor:
        x = self.conv1(x)
        if keepDim:
            x = torch.nn.functional.pad(x, [3, 3, 3, 3], value=float('-inf'))

        x = self.bn1(x)
        x = self.relu(x)
        x, _ = self.layer1((x, keepDim))
        x, _ = self.layer2((x, keepDim))
        x, _ = self.layer3((x, keepDim))
        x, _ = self.layer4((x, keepDim))
        return x
