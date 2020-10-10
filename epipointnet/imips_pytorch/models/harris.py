from typing import Optional

import kornia
import torch

from . import pyramid


class AvgScaleHarris(torch.nn.Module):

    def __init__(self, max_octaves: Optional[int] = None,
                 scales_per_octave: Optional[int] = 3,
                 ):
        super(AvgScaleHarris, self).__init__()

        self.scale_pyramid = pyramid.ScalePyramid(
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
