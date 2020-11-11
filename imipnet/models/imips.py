import abc
from typing import Optional, Union

import torch
import torch.nn.functional


class ImipNet(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, input_channels: int, output_channels: int):
        super(ImipNet, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels

    def input_channels(self) -> int:
        return self._input_channels

    def output_channels(self) -> int:
        return self._output_channels

    def receptive_field_diameter(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def extract_keypoints(self, image: torch.Tensor, exclude_border_px: Optional[int] = None) -> (
            torch.Tensor, torch.Tensor):
        defer_set_train = False
        if self.training:
            self.train(False)
            defer_set_train = True

        # Only return keypoints for which there is valid responses
        if exclude_border_px is None or exclude_border_px < (self.receptive_field_diameter() - 1) // 2:
            exclude_border_px = (self.receptive_field_diameter() - 1) // 2

        # assume image is CxHxW
        assert len(image.shape) == 3 and image.shape[0] == self._input_channels

        # add the batch dimension
        image = image.unsqueeze(0)

        # output: 1xCxHxW -> CxHxW
        output: torch.Tensor = self.__call__(image, keepDim=True).squeeze(dim=0)

        # TODO: test removing the exclusion now that it is handled in the model implementation
        # don't return any keypoint that is in the receptive field radius
        # from the border
        output[:, :exclude_border_px, :] = float('-inf')
        output[:, :, :exclude_border_px] = float('-inf')
        output[:, -exclude_border_px:, :] = float('-inf')
        output[:, :, -exclude_border_px:] = float('-inf')

        # CxHxW -> CxI where I = H*W
        output_shape = output.shape
        output_linear = output.reshape((output_shape[0], -1))

        # C length
        linear_arg_maxes = output_linear.argmax(dim=1)

        # 2xC, x_pos = linear mod width, y_pos = linear / width
        keypoints_xy = torch.zeros((2, output_shape[0]), device=output.device)
        keypoints_xy[0, :] = linear_arg_maxes % output_shape[2]
        keypoints_xy[1, :] = linear_arg_maxes // output_shape[2]

        if defer_set_train:
            self.train(True)
        return keypoints_xy, output

    @torch.no_grad()
    def extract_top_k_keypoints(self, img: torch.Tensor, k: int):
        # assume image is CxHxW
        assert len(img.shape) == 3 and img.shape[0] == self.input_channels()

        defer_set_train = False
        if self.training:
            self.train(False)
            defer_set_train = True

        output: torch.Tensor = self.__call__(img.unsqueeze(0), keepDim=True)

        # non maxima suppression
        output_nms_mask = ~(output == torch.nn.functional.max_pool2d(output, 3, stride=1, padding=1))

        # HACK: set the negative outputs channel by channel to avoid 4*RAM usage from indexing
        # see: https://github.com/pytorch/pytorch/issues/30246
        for c in range(output.shape[1]):
            output[0, c][output_nms_mask[0, c]] = float("-inf")

        output = output.squeeze(dim=0).flatten(1)  # C x HW

        topk_scores, topk_keypoints = torch.topk(output, k, -1, largest=True, sorted=True)  # C x K

        keypoints_2ck = torch.zeros((2, output.shape[0], k), device=output.device, dtype=topk_scores.dtype)
        # return values in x, y format
        keypoints_2ck[0, :, :] = topk_keypoints % img.shape[2]
        keypoints_2ck[1, :, :] = topk_keypoints // img.shape[2]

        if defer_set_train:
            self.train(True)

        return keypoints_2ck, topk_scores

    def forward(self, images: torch.Tensor, keepDim: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def regularizer(self) -> Union[torch.Tensor, int, float]:
        return 0
