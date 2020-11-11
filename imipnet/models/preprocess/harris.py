from typing import Optional

import kornia
import torch

from .preprocess import PreprocessModule
from .pyramid import ScalePyramid


class PreprocessHarris(PreprocessModule):
    def __init__(self):
        super(PreprocessModule, self).__init__()
        self.harris = AvgScaleHarris(
            max_octaves=2,
            scales_per_octave=3,
        )

    def preprocess(self, image: torch.Tensor):
        # normalize image and create harris corners
        image = (image / 127.5) - 1.0  # scale to [-1, 1]

        # convert to grayscale
        image_gray = kornia.color.rgb_to_grayscale(image) if image.shape[0] == 3 else image

        # contrast stretch image for harris http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
        image_gray_1p_idx = 1 + round(.02 * (image_gray.numel() - 1))
        image_gray_99p_idx = 1 + round(.98 * (image_gray.numel() - 1))
        image_gray_1p_val = image_gray.view(-1).kthvalue(image_gray_1p_idx).values
        image_gray_99p_val = image_gray.view(-1).kthvalue(image_gray_99p_idx).values
        image_gray = (image_gray - image_gray_1p_val) * (2.0 / (image_gray_99p_val - image_gray_1p_val)) - 1
        image_gray.clamp_(-1, 1)
        image_harris = self.harris(image_gray.unsqueeze(0))[0]
        return torch.cat((image, image_harris), dim=0)

    def output_channels(self, input_channels: int) -> int:
        return input_channels + 1


class AvgScaleHarris(torch.nn.Module):

    def __init__(self, max_octaves: Optional[int] = None,
                 scales_per_octave: Optional[int] = 3,
                 ):
        super(AvgScaleHarris, self).__init__()

        self.scale_pyramid = ScalePyramid(
            max_octaves=max_octaves,
            n_levels=scales_per_octave,
            init_sigma=1.6,
            double_image=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pyramid, sigmas, _ = self.scale_pyramid(images)
        ret_harris = torch.zeros_like(images)
        for oct_idx, octave in enumerate(pyramid):
            B, L, CH, H, W = octave.size()  # L dim is the levels per octave
            harris_response = kornia.feature.harris_response(
                octave.view(B * L, CH, H, W), grads_mode='sobel', sigmas=sigmas[oct_idx].view(-1)
            )
            harris_response = torch.nn.functional.interpolate(
                harris_response,
                size=(images.shape[2], images.shape[3]),
                mode='bilinear', align_corners=False
            ).view(B, L, CH, images.shape[2], images.shape[3]).sum(dim=1)
            ret_harris = (ret_harris + harris_response)
            pyramid[oct_idx] = None  # drop the reference to the scaled image for GC
            sigmas[oct_idx] = None

        ret_harris = ret_harris / (len(pyramid) * self.scale_pyramid.n_levels)
        ret_harris = ret_harris.abs().sqrt() * ret_harris.sign()
        return ret_harris
