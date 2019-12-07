import abc

import torch
import torch.nn.functional


class ImipsNet(torch.nn.Module, abc.ABC):
    def __init__(self, input_channels: int = 1):
        super(ImipsNet, self).__init__()
        self._input_channels = input_channels

    def input_channels(self) -> int:
        return self._input_channels

    def receptive_field_diameter(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def extract_keypoints(self, image: torch.Tensor) -> torch.Tensor:
        # assume image is HxW or HxWx3

        # Add a singleton dimension for channels if it doesn't exist
        if len(image.shape) == 2 and self._input_channels == 1:
            image = image.unsqueeze(0)
        # Change format from HWC to CHW for color images
        elif len(image.shape) == 3 and self._input_channels == 3:
            image = image.permute(2, 0, 1)

        assert len(image.shape) == 3 and image.shape[2] == self._input_channels

        # add the batch dimension
        image = image.unsqueeze(0)

        # output: 1xCxHxW
        output: torch.Tensor = self.__call__(image, True)

        # 1xHxWxC -> HxWxC -> IxC where I = H*W
        output = output.squeeze(dim=0)
        output_shape = output.shape

        output = output.reshape((-1, output_shape[2]))

        # C length
        linear_arg_maxes = output.argmax(dim=0)

        # 2xC, x_pos = linear mod height, y_pos = linear / height
        keypoints_xy = torch.zeros((2, output_shape[2]))
        keypoints_xy[0, :] = linear_arg_maxes % output_shape[1]
        keypoints_xy[1, :] = linear_arg_maxes // output_shape[1]
        return keypoints_xy

    def forward(self, images: torch.Tensor, keepDim: bool) -> torch.Tensor:
        raise NotImplementedError
