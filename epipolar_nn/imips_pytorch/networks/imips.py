import abc

import torch
import torch.nn.functional


class ImipsNet(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, input_channels: int, output_channels: int):
        super(ImipsNet, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels

    def input_channels(self) -> int:
        return self._input_channels

    def output_channels(self) -> int:
        return self._output_channels

    def receptive_field_diameter(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def extract_keypoints(self, image: torch.Tensor) -> torch.Tensor:
        # assume image is CxHxW
        assert len(image.shape) == 3 and image.shape[0] == self._input_channels

        # add the batch dimension
        image = image.unsqueeze(0)

        # output: 1xCxHxW
        output: torch.Tensor = self.__call__(image, True)

        # 1xCxHxW -> CxHxW -> CxI where I = H*W
        output = output.squeeze(dim=0)
        output_shape = output.shape

        output = output.reshape((output_shape[0], -1))

        # C length
        linear_arg_maxes = output.argmax(dim=1)

        # 2xC, x_pos = linear mod width, y_pos = linear / width
        keypoints_xy = torch.zeros((2, output_shape[0]), device="cuda")
        keypoints_xy[0, :] = linear_arg_maxes % output_shape[2]
        keypoints_xy[1, :] = linear_arg_maxes // output_shape[2]
        return keypoints_xy

    def forward(self, images: torch.Tensor, keepDim: bool) -> torch.Tensor:
        raise NotImplementedError
